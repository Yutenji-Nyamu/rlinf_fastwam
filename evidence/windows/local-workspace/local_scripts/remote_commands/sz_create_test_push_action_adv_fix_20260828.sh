#!/usr/bin/env bash
set -euo pipefail

SOURCE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv-fix
PATCH=/tmp/sz_action_adv_fix_20260828.patch
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
BASE=a5b94b6f10a9212502d6930f07543f61e31af52e
BRANCH=codex/sz-grpo-dvac-action-adv-fix

test "$(git -C "$SOURCE" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$SOURCE" status --short)"
test -s "$PATCH"
if [[ -e "$TARGET" ]]; then
  test "$(git -C "$TARGET" rev-parse HEAD)" = "$BASE"
  test "$(git -C "$TARGET" branch --show-current)" = "$BRANCH"
  test -z "$(git -C "$TARGET" status --short)"
else
  test -z "$(git -C "$SOURCE" branch --list "$BRANCH")"
  git -C "$SOURCE" worktree add -b "$BRANCH" "$TARGET" "$BASE"
fi
git -C "$TARGET" apply --check "$PATCH"
git -C "$TARGET" apply "$PATCH"

cd "$TARGET"
expected=$'rlinf/algorithms/losses.py\nrlinf/workers/actor/embodied_fsdp_actor_worker.py\ntests/unit_tests/test_dvac_train_weighting.py'
test "$(git diff --name-only | sort)" = "$expected"
git diff --check

source "$VENV/bin/activate"
export REPO_PATH="$TARGET" EMBODIED_PATH="$TARGET/examples/embodiment"
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$TARGET:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1

ruff check \
  rlinf/algorithms/losses.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  tests/unit_tests/test_dvac_train_weighting.py
ruff format --check \
  rlinf/algorithms/losses.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  tests/unit_tests/test_dvac_train_weighting.py
pytest -q tests/unit_tests/test_dvac_train_weighting.py

python examples/embodiment/train_embodied_agent.py \
  --config-path "$TARGET/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  algorithm.logprob_type=action_level \
  algorithm.dvac_gradient_weighting.mode=apply \
  algorithm.dvac_gradient_weighting.application=action_advantage \
  algorithm.dvac_gradient_weighting.weight_min=0.0 \
  algorithm.dvac_gradient_weighting.weight_max=2.0 \
  --cfg job --resolve >/tmp/sz-action-adv-fix-resolved-20260828.yaml
grep -q '^  logprob_type: action_level$' /tmp/sz-action-adv-fix-resolved-20260828.yaml
grep -q '^    application: action_advantage$' /tmp/sz-action-adv-fix-resolved-20260828.yaml
grep -q '^    weight_min: 0.0$' /tmp/sz-action-adv-fix-resolved-20260828.yaml
grep -q '^    weight_max: 2.0$' /tmp/sz-action-adv-fix-resolved-20260828.yaml

git add \
  rlinf/algorithms/losses.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  tests/unit_tests/test_dvac_train_weighting.py
git commit -m 'fix(embodiment): preserve action-advantage update scale'
git push personal HEAD:refs/heads/"$BRANCH"

printf 'FIX_HEAD=%s\n' "$(git rev-parse HEAD)"
git status --short
git ls-remote personal refs/heads/"$BRANCH"
echo SZ_ACTION_ADV_FIX_IMPLEMENTED_TESTED_PUSHED
