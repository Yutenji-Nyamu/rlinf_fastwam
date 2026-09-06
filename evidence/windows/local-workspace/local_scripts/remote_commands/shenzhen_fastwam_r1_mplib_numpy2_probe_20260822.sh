#!/usr/bin/env bash
set -euo pipefail

ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa
PY="$ENV/bin/python"

test -x "$PY"
source /etc/profile.d/mihomo-proxy.sh
export PATH="$ENV/bin:$PATH"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="$CACHE_ROOT/tmp"
unset PYTHONPATH

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
"$PY" - <<'PY'
import numpy
assert numpy.__version__ == '2.2.6'
print({'numpy_before': numpy.__version__})
PY
"$PY" -m pip install --index-url https://pypi.org/simple --no-deps 'mplib==0.2.1'
"$PY" - <<'PY'
import importlib.metadata as md
from pathlib import Path
import mplib
import mplib.planner
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import numpy

assert md.version('mplib') == '0.2.1'
assert numpy.__version__ == '2.2.6'
assert Path(mplib.__file__).resolve().is_file()
print({
    'mplib': md.version('mplib'),
    'mplib_file': mplib.__file__,
    'numpy': numpy.__version__,
    'planner_symbols': [SapienPlanner.__name__, SapienPlanningWorld.__name__],
})
PY
check_output="$($PY -m pip check 2>&1 || true)"
printf '%s\n' "$check_output"
test "$check_output" = 'mplib 0.2.1 has requirement numpy<2.0, but you have numpy 2.2.6.'
printf '%s\n' FASTWAM_MPLIB_NUMPY2_IMPORT_OK_WITH_KNOWN_METADATA_MISMATCH
