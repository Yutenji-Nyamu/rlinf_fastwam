# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Seal completed CPU checks and launch the accepted exact-owner handoff."""

import json
import subprocess
import sys
from pathlib import Path

from prepare import BASE, read, save

for method, gpus in [("dsrl", ["4", "5"]), ("rlt", ["6"])]:
    result = read(BASE / f"test-changes-{method}.json")
    assert result and all(v for v in result.values())
    rows = {}
    for g in gpus:
        record = read(BASE / f"runtime-check-{method}-g{g}.json")
        assert record["result"] == "pass"
        rows.update(record["checks"])
    save(BASE / f"runtime-check-{method}.json", {"result": "pass", "checks": rows})
assert not (BASE / "launch-receipt.json").exists()
with (BASE / "launch.log").open("x") as f:
    p = subprocess.Popen(
        [sys.executable, "-u", "-B", str(Path(__file__).parent / "launch.py")],
        stdin=subprocess.DEVNULL,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(json.dumps({"launch_pid": p.pid, "base": str(BASE)}))
