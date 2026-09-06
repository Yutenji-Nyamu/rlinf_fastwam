#!/usr/bin/env bash
set -euo pipefail

path=/root/autodl-tmp/RoboTwin/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
nl -ba "$path" | sed -n '180,360p'

printf '%s\n' '=== related_environment ==='
cd /root/autodl-tmp/RoboTwin/policy/pi0
printf 'HF_LEROBOT_HOME=%s\n' "${HF_LEROBOT_HOME:-UNSET}"
printf 'HF_HOME=%s\n' "${HF_HOME:-UNSET}"
if [[ -x .venv/bin/python ]]; then
  readlink -f .venv/bin/python
  .venv/bin/python -B -c \
    'import sys, lerobot, h5py, cv2; print(sys.version); print("lerobot", getattr(lerobot, "__version__", "unknown")); print("h5py", h5py.__version__); print("cv2", cv2.__version__)'
else
  printf '%s\n' 'MISSING policy/pi0/.venv'
fi
