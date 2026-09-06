#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

BASE=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
ENV=/home/chenyiteng/venvs/rlt-data-convert-lerobot010
CONVERTER=/data/chenyiteng/datasets/robotwin2/tooling/RoboTwin-c3ddfa8b97d5519efa828b075999bd0006778e5e/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py

test ! -e "$ENV"
"$BASE" -m venv --system-site-packages "$ENV"
"$ENV/bin/python" -m pip install --no-deps 'lerobot==0.1.0'

CONVERTER="$CONVERTER" "$ENV/bin/python" -B - <<'PY'
import importlib.util
import os
import lerobot
spec = importlib.util.spec_from_file_location("robotwin_lerobot_converter", os.environ["CONVERTER"])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print("LEROBOT_VERSION", getattr(lerobot, "__version__", "unknown"))
print("CONVERTER_IMPORT_OK", callable(module.create_empty_dataset), callable(module.populate_dataset))
PY
