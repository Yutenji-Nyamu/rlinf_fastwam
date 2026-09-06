#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

ENV=/home/chenyiteng/venvs/rlt-data-convert-lerobot010
CONVERTER=/data/chenyiteng/datasets/robotwin2/tooling/RoboTwin-c3ddfa8b97d5519efa828b075999bd0006778e5e/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py

test -x "$ENV/bin/python"
"$ENV/bin/python" -m pip install --no-deps --force-reinstall \
  'git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'

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
