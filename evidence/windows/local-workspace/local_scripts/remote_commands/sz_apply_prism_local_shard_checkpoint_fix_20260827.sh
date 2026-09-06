#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
PATCH=/tmp/sz-prism-local-shard-checkpoint-20260827.diff
BRANCH=codex/sz-prism-dvac-rank-rloo
BEFORE=3f977ce7f0c9271a4e164b4d602166286a9ad9f3

test "$(git -C "$WT" rev-parse HEAD)" = "$BEFORE"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$WT" status --short)"
test -s "$PATCH"
git -C "$WT" apply --check "$PATCH"
git -C "$WT" apply "$PATCH"

source "$VENV/bin/activate"
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export ROBOTWIN_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$WT:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1

cd "$WT"
git diff --check
python -m py_compile rlinf/hybrid_engines/fsdp/fsdp_model_manager.py
python examples/embodiment/train_embodied_agent.py \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  +actor.fsdp_config.checkpoint_format=local_shard \
  --cfg job --resolve > /tmp/sz-prism-local-shard-compose-20260827.yaml
grep -A24 '^  fsdp_config:' /tmp/sz-prism-local-shard-compose-20260827.yaml | \
  grep -F 'checkpoint_format: local_shard'

git add -- rlinf/hybrid_engines/fsdp/fsdp_model_manager.py
git diff --cached --check
git commit -m "fix: allow opting FSDP checkpoints into local shards"
git push personal "$BRANCH"
printf 'before=%s\nafter=%s\n' "$BEFORE" "$(git rev-parse HEAD)"
git status --short
echo SZ_PRISM_LOCAL_SHARD_CHECKPOINT_FIX_PUSHED
