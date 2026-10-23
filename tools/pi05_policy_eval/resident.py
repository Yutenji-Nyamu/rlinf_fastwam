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
