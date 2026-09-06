#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
RUN=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
source "$VENV/bin/activate"
export PYTHONPATH="$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"

exec > >(tee "$RUN/curobo_contract_probe.log") 2>&1

printf '%s\n' '=== INSTALLED CUROBO PACKAGE ==='
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import curobo

dist = md.distribution("nvidia-curobo")
print({"version": dist.version, "curobo_file": curobo.__file__, "curobo_path": list(curobo.__path__)})
direct = Path(dist._path) / "direct_url.json"
print(direct.read_text() if direct.is_file() else "NO direct_url.json")
print({"types_attr": repr(getattr(curobo, "types", None))})
PY

find "$VENV/lib/python3.11/site-packages/curobo" -maxdepth 3 -printf '%y %P\n' | sort | head -n 240

printf '%s\n' '=== COMPATIBILITY IMPORT CONTRACT ==='
grep -RIn --include='*.py' 'from curobo\.types\|import curobo\.types' "$ROBOTWIN" | head -n 120 || true

printf '%s\n' '=== TARGETED IMPORTS ==='
set +e
"$VENV/bin/python" - <<'PY'
from curobo.types.base import TensorDeviceType
print(TensorDeviceType)
PY
base_rc=$?
"$VENV/bin/python" - <<'PY'
from robotwin.envs import vector_env
print(vector_env.__file__)
PY
vector_env_rc=$?
set -e
printf 'curobo_types_base_exit=%s\n' "$base_rc"
printf 'robotwin_vector_env_exit=%s\n' "$vector_env_rc"
