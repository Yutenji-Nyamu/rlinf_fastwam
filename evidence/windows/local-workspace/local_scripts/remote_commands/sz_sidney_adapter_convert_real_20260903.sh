#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
SRC=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
OUT=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
cd "$WT"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
"$PY" -m rlinf.utils.ckpt_convertor.openpi.convert \
  --mode lerobot_pi05_to_openpi_rlinf \
  --input-model "$SRC" \
  --output-model "$OUT" \
  --source-revision e49e2ab6c11f07511573b67261bd129e88d0a416
