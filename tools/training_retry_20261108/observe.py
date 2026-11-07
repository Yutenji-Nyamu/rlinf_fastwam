# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Read the four retry owners, progress files and physical GPU assignments."""

import json
import sys
import time
from pathlib import Path

from prepare import BASE, SCHEDULER, read

sys.path.insert(0, str(SCHEDULER / "ops"))
from common import gpus, same


def optional(path):
    return read(path) if path.is_file() else None


def tail(path, limit=3500):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, stream.seek(0, 2) - limit))
        return stream.read().decode(errors="replace")


def main():
    entries = read(BASE / "prepared-dsrl.json")["entries"]
    out = {
        "time": time.time(),
        "launch": optional(BASE / "launch-status.json"),
        "gpus": gpus(),
        "runs": {},
        "scheduler": read(SCHEDULER / "status.json"),
        "join": read(
            "/data/chenyiteng/deployment-20261104/parallel-u47/join-status.json"
        ),
    }
    for g, e in entries.items():
        d = Path(e["deployment"])
        identity = optional(d / "identity.json")
        row = {
            "identity": identity,
            "alive": same(identity) if identity else False,
            "status": optional(d / "status.json"),
            "reference": optional(d / "reference-eval.json"),
            "smoke_complete": optional(d / "smoke-complete.json"),
            "logs": {},
        }
        for name in ["outer", "reference_eval", "smoke", "formal", f"native-g{g}"]:
            t = tail(d / f"{name}.log")
            if t:
                row["logs"][name] = t
        cps = Path(e["run"]) / "formal/formal/checkpoints"
        row["checkpoint_save_markers"] = [
            p.parent.name for p in cps.glob("global_step_*/dojo-complete.json")
        ]
        out["runs"][g] = row
    out["launch_log"] = tail(BASE / "launch.log")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
