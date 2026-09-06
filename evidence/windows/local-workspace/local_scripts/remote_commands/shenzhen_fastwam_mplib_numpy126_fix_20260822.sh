#!/usr/bin/env bash
set -euo pipefail

ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
PYTHON="$ENV/bin/python"

source /etc/profile.d/mihomo-proxy.sh
export PIP_CACHE_DIR=/home/chenyiteng/cache/fastwam-7faa/pip
"$PYTHON" -m pip install --no-deps --force-reinstall 'numpy==1.26.4'
"$PYTHON" - <<'PY'
from importlib.metadata import version
print("numpy=" + version("numpy"))
print("mplib=" + version("mplib"))
PY
