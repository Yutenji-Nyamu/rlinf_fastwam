# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Read-only short snapshot during the three-card cutover."""

import json
import os
import subprocess
import time
from pathlib import Path

from prepare import ACTIVE, BASE, read

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(p):
    try:
        return read(p)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def curves(root):
    result = {}
    for p in {p.parent for p in Path(root).rglob("events.out.tfevents.*")}:
        a = EventAccumulator(str(p), size_guidance={"scalars": 0})
        a.Reload()
        result[str(p)] = {
            key: [[v.step, v.value, v.wall_time] for v in a.Scalars(key)]
            for key in a.Tags()["scalars"]
        }
    return result


out = {"time": time.time(), "launch": load(BASE / "launch-status.json"), "slots": {}}
for g, e in read(ACTIVE)["slots"].items():
    d = Path(e["owner"])
    status = load(d / "status.json")
    identity = load(d / "identity.json")
    row = {
        "status": status,
        "identity": identity,
        "alive": False,
    }
    if identity:
        try:
            p = Path("/proc", str(identity["pid"]))
            stat = (p / "stat").read_text().rsplit(") ", 1)[1].split()
            row["alive"] = (
                p.stat().st_uid == identity["uid"]
                and int(stat[19]) == identity["start"]
            )
        except FileNotFoundError:
            pass
    for n in ["outer.log", "smoke.log", "reference_eval.log", "formal.log"]:
        f = d / n
        if f.exists():
            with f.open("rb") as b:
                b.seek(max(0, f.stat().st_size - 2200))
                row[n] = b.read().decode(errors="replace")
            row[n + "_mtime"] = f.stat().st_mtime
    row["receipts"] = {p.name: load(p) for p in d.glob("*-complete.json")}
    row["native"] = {str(p): load(p) for p in d.glob("native-g*/progress.json")}
    row["run"] = e["run"]
    row["streams"] = curves(e["run"])
    row["best"] = load(Path(e["run"]) / "best-evaluation.json")
    out["slots"][g] = row
out["gpu"] = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
    timeout=25,
).stdout
out["disk"] = subprocess.run(
    ["df", "-B1", "/home", "/data", "/data2"], capture_output=True, text=True
).stdout
out["scheduler"] = load(
    "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/status.json"
)
latest_launch = max(
    BASE.glob("launch*.log"), key=lambda p: p.stat().st_mtime, default=None
)
out["launch_tail"] = latest_launch.read_text()[-3000:] if latest_launch else ""
print(json.dumps(out))
