#!/usr/bin/env bash
set -euo pipefail
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$root"

echo '=== openpi builder ==='
sed -n '1,180p' rlinf/models/embodiment/openpi/__init__.py
echo '=== action model predict/sample ==='
sed -n '820,1105p' rlinf/models/embodiment/openpi/openpi_action_model.py
echo '=== robotwin env init/reset/obs ==='
sed -n '1,330p' rlinf/envs/robotwin/robotwin_env.py
echo '=== env factory ==='
grep -RInE 'RoboTwinEnv\(|register.*robotwin|env_type.*robotwin|build_env|make_env' rlinf/envs rlinf/workers/env | sed -n '1,220p'
echo '=== worker model build and model obs path ==='
sed -n '120,350p' rlinf/workers/rollout/hf/huggingface_worker.py
sed -n '430,650p' rlinf/workers/rollout/hf/huggingface_worker.py
echo 'GPU_RAY_MODEL_SIM_USED=0'
