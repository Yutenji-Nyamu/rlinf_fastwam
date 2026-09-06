#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
BASE=0e28ac6f09f821ea12e7d54eba7118ce0000ca86

test "$(git -C "$WT" rev-parse HEAD)" = "$BASE"
test "$(git -C "$WT" branch --show-current)" = codex/sz-grpo-dvac-action-adv
cd "$WT"
expected=$'examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml\nrlinf/algorithms/utils.py\nrlinf/workers/actor/embodied_fsdp_actor_worker.py\ntests/unit_tests/test_dvac_train_weighting.py'
test "$(git diff --name-only | sort)" = "$expected"

source "$VENV/bin/activate"
export REPO_PATH="$WT" EMBODIED_PATH="$WT/examples/embodiment"
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1

git diff --check
ruff check \
  rlinf/algorithms/utils.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  tests/unit_tests/test_dvac_train_weighting.py
ruff format --check \
  rlinf/algorithms/utils.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  tests/unit_tests/test_dvac_train_weighting.py
python - <<'PY'
from pathlib import Path
for name in (
    "rlinf/algorithms/utils.py",
    "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
    "tests/unit_tests/test_dvac_train_weighting.py",
):
    path = Path(name)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("SZ_ACTION_ADV_COMPILE_OK")
PY
pytest -q tests/unit_tests/test_dvac_train_weighting.py

python examples/embodiment/train_embodied_agent.py \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  --cfg job --resolve > /tmp/sz-action-adv-control-resolved-20260827.yaml
python examples/embodiment/train_embodied_agent.py \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  algorithm.logprob_type=action_level \
  algorithm.dvac_gradient_weighting.mode=apply \
  algorithm.dvac_gradient_weighting.application=action_advantage \
  algorithm.dvac_gradient_weighting.weight_min=0.0 \
  algorithm.dvac_gradient_weighting.weight_max=2.0 \
  --cfg job --resolve > /tmp/sz-action-adv-method-resolved-20260827.yaml

grep -q '^  logprob_type: chunk_level$' /tmp/sz-action-adv-control-resolved-20260827.yaml
grep -q '^    application: logprob_st$' /tmp/sz-action-adv-control-resolved-20260827.yaml
grep -q '^  logprob_type: action_level$' /tmp/sz-action-adv-method-resolved-20260827.yaml
grep -q '^    mode: apply$' /tmp/sz-action-adv-method-resolved-20260827.yaml
grep -q '^    application: action_advantage$' /tmp/sz-action-adv-method-resolved-20260827.yaml
grep -q '^    weight_min: 0.0$' /tmp/sz-action-adv-method-resolved-20260827.yaml
grep -q '^    weight_max: 2.0$' /tmp/sz-action-adv-method-resolved-20260827.yaml

git diff --stat
git status --short
echo SZ_ACTION_ADV_MINIMAL_CHECKS_OK
