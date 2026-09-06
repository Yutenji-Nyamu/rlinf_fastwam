#!/usr/bin/env bash
set -euo pipefail

FWPY=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
NATIVEPY=/home/chenyiteng/miniforge3/envs/RoboTwin/bin/python
CONVERSION=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/lib/python3.10/site-packages/mplib/sapien_utils/conversion.py

printf '%s\n' '=== crashing source line ==='
nl -ba "$CONVERSION" | sed -n '292,318p'
printf '%s\n' '=== fastwam versions ==='
"$FWPY" - <<'PY'
from importlib.metadata import version
for name in ("numpy", "mplib", "sapien", "scipy", "fastwam"):
    print(name, version(name))
PY
printf '%s\n' '=== native working versions ==='
"$NATIVEPY" - <<'PY'
from importlib.metadata import version
for name in ("numpy", "mplib", "sapien", "scipy"):
    print(name, version(name))
PY
