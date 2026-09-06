#!/usr/bin/env bash
set -euo pipefail
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$root"
sed -n '420,455p' rlinf/workers/env/env_worker.py
sed -n '1000,1040p' rlinf/workers/rollout/hf/huggingface_worker.py
sed -n '1035,1115p' rlinf/models/embodiment/openpi/openpi_action_model.py
sed -n '1115,1155p' rlinf/models/embodiment/openpi/openpi_action_model.py
echo 'GPU_RAY_MODEL_SIM_USED=0'
