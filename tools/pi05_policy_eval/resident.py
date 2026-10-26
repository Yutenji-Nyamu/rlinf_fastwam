# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Keep one frozen model per task slot; exit on any failed task for clean restart."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from protocol import read, save
from run_batch import run


def repeated_collection_failure(config):
    """Do not rerun a collection task after two matching infrastructure failures."""
    path = config.get("collection_failure_log")
    if not config.get("collect_transitions") or not path or not Path(path).is_file():
        return False
    rows = [
        r for r in read(path).get("results", [])
        if r["task"].startswith(config["task"] + "-b")
        and "collection task quarantined" not in (r.get("error") or "")
    ]
    if len(rows) < 2 or any(r["valid"] for r in rows[-2:]):
        return False
    errors = [(r.get("error") or "").strip().splitlines() for r in rows[-2:]]
    return bool(errors[0] and errors[1] and errors[0][-1] == errors[1][-1])


def main(status_path):
    """Consume one config per line; publish idle only after environment cleanup."""
    cache = {}
    save(status_path, {"state": "ready", "time": time.time(), "pid": os.getpid()})
    for line in sys.stdin:
        command = json.loads(line)
        if command.get("stop"):
            return
        config = command["config"]
        c = read(config)
        save(status_path, {"state": "busy", "output": c["output"], "time": time.time()})
        try:
            if repeated_collection_failure(c):
                raise RuntimeError(
                    "collection task quarantined after two matching failures"
                )
            run(config, model_cache=cache)
        except BaseException:
            save(
                Path(c["output"]) / "error.json",
                {"time": time.time(), "error": traceback.format_exc()},
            )
            raise
        save(status_path, {"state": "idle", "output": c["output"], "time": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("status")
    main(parser.parse_args().status)
