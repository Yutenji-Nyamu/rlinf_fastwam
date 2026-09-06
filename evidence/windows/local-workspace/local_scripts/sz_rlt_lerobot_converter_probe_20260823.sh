#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CONVERTER=/data/chenyiteng/datasets/robotwin2/tooling/RoboTwin-c3ddfa8b97d5519efa828b075999bd0006778e5e/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
PY=/home/chenyiteng/venvs/rlt-data-convert-lerobot010/bin/python
CONVERTER="$CONVERTER" "$PY" -B - <<'PY'
import importlib.util
import os
spec = importlib.util.spec_from_file_location("robotwin_lerobot_converter", os.environ["CONVERTER"])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print("IMPORT_OK")
print("HAS_CREATE", callable(module.create_empty_dataset))
print("HAS_POPULATE", callable(module.populate_dataset))
PY
