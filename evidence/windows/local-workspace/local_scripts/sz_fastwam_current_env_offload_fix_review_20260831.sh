#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

test "$(git -C "$WT" rev-parse HEAD)" = 81076b13e91a3db0b8def6b48dd6e1db2897cb2b
test "$(git -C "$WT" status --short | wc -l)" -eq 1
test "$(git -C "$WT" status --short)" = ' M examples/embodiment/config/robotwin_move_stapler_pad_grpo_fastwam.yaml'
git -C "$WT" diff --check
git -C "$WT" diff --unified=3 -- examples/embodiment/config/robotwin_move_stapler_pad_grpo_fastwam.yaml

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389 ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FW:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
resolved=$(mktemp)
trap 'rm -f "$resolved"' EXIT
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_move_stapler_pad_grpo_fastwam --cfg job --resolve > "$resolved"
"$VENV/bin/python" - "$resolved" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
assert cfg['env']['train']['enable_offload'] is True
assert cfg['env']['eval']['enable_offload'] is True
assert cfg['rollout']['enable_offload'] is True
assert cfg['actor']['enable_offload'] is False
assert cfg['env']['train']['total_num_envs'] == 32
assert cfg['env']['train']['rollout_epoch'] == 4
assert cfg['algorithm']['group_size'] == 8
assert cfg['actor']['global_batch_size'] == 1024
print('ENV_OFFLOAD_FIX_COMPOSE_OK')
PY
