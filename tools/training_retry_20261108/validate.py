# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Run config/head/scheduler checks on CPU, one Cluster singleton per process."""

import os
import subprocess
import sys

from prepare import BASE, EXPORT, read, save


def main():
    package = BASE / "package"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [sys.executable, "-B", str(package / "check_prepare.py")], env=env, check=True
    )
    for method, gpus in [("rlt", ["6", "7"]), ("dsrl", ["4", "5"])]:
        rows = {}
        for gpu in gpus:
            subprocess.run(
                [sys.executable, "-B", str(package / "check_runtime.py"), method, gpu],
                cwd=EXPORT / method,
                env=dict(env, PYTHONPATH=str(EXPORT / method)),
                check=True,
            )
            rows.update(read(BASE / f"runtime-check-{method}-g{gpu}.json")["checks"])
        save(BASE / f"runtime-check-{method}.json", {"result": "pass", "checks": rows})


if __name__ == "__main__":
    main()
