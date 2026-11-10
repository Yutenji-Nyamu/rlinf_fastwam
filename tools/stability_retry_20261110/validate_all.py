# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
import os
import subprocess
import sys
from pathlib import Path

from prepare import BASE, EXPORT, read, save

P = Path(__file__).parent
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="2",
    PYTHONDONTWRITEBYTECODE="1",
)
for method, gpus in [("dsrl", ["4", "5"]), ("rlt", ["6"])]:
    rows = {}
    for g in gpus:
        subprocess.run(
            [sys.executable, "-B", str(P / "check_runtime.py"), method, g],
            env=dict(env, PYTHONPATH=str(EXPORT / method)),
            cwd=EXPORT / method,
            check=True,
        )
        rows.update(read(BASE / f"runtime-check-{method}-g{g}.json")["checks"])
    subprocess.run(
        [sys.executable, "-B", str(P / "test_changes.py"), method],
        env=dict(env, PYTHONPATH=str(EXPORT / method)),
        cwd=EXPORT / method,
        check=True,
    )
    save(BASE / f"runtime-check-{method}.json", {"result": "pass", "checks": rows})
print("ALL_CPU_CHECKS_PASSED")
