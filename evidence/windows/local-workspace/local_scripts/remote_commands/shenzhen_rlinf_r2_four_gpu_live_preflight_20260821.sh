#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
SFT_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1
PPO_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1
RELOAD_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-reload-fixed64-4gpu4567-v1

test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test ! -e "$SFT_RUN"
test ! -e "$PPO_RUN"
test ! -e "$RELOAD_RUN"

unset CUDA_VISIBLE_DEVICES
source "$VENV/bin/activate"
"$VENV/bin/python" - <<'PY'
import os
from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager as N
print({
    "driver_CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
    "node_gpu_count": N.get_current_node_num_accelerators(),
    "driver_visible_gpu_ids": N.get_current_process_visible_accelerator_ids(),
})
assert os.getenv("CUDA_VISIBLE_DEVICES") is None
assert N.get_current_node_num_accelerators() == 8
assert N.get_current_process_visible_accelerator_ids() is None
PY

printf '%s\n' '=== all GPUs ==='
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
printf '%s\n' '=== compute apps on physical GPU 4-7 ==='
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader,nounits || true
printf '%s\n' '=== existing Ray control processes ==='
pgrep -a -x raylet || true
pgrep -a -x gcs_server || true

if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'target GPU 4-7 are occupied' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  printf '%s\n' 'existing Ray control process found; do not start a second RLinf cluster' >&2
  exit 1
fi

df -h /home /data
printf '%s\n' 'R2_FOUR_GPU_LIVE_PREFLIGHT_OK'
