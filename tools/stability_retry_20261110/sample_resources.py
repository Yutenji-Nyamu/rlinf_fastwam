# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Bounded read-only sample during real jobs; no CUDA context."""

import json
import subprocess
import time
from pathlib import Path

base = Path("/data/chenyiteng/deployment-20261110/stability-retry")
active = Path("/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json")
with (base / "resource-samples.jsonl").open("x") as f:
    for _ in range(120):
        row = {"time": time.time()}
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            row["gpus"] = [
                list(map(int, line.split(","))) for line in r.stdout.splitlines()
            ]
            slots = json.loads(active.read_text())["slots"]
            row["phases"] = {
                g: json.loads((Path(v["owner"]) / "status.json").read_text()).get(
                    "phase"
                )
                for g, v in slots.items()
            }
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            row["error"] = str(exc)
        f.write(json.dumps(row) + "\n")
        f.flush()
        time.sleep(10)
