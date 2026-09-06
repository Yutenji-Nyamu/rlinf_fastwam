set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
CHILD="$ROOT/examples/embodiment/config/robotwin_adjust_bottle_prism_dvac_rank_rloo_openpi.yaml"
EXPECTED=0e28ac6f09f821ea12e7d54eba7118ce0000ca86

test "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED"
if test -e "$CHILD"; then
  child_status=$(git -C "$ROOT" status --short -- "$CHILD")
  test "$child_status" = "?? examples/embodiment/config/robotwin_adjust_bottle_prism_dvac_rank_rloo_openpi.yaml"
  rm -- "$CHILD"
fi

source /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/activate
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROOT/examples/embodiment"
export ROBOTWIN_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$ROOT:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''

cd "$ROOT"
git diff --check
ruff check \
  rlinf/algorithms/advantages.py \
  rlinf/algorithms/dvac_rank_reward.py \
  rlinf/config.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_prism_dvac_rank_rloo.py
ruff format --check \
  rlinf/algorithms/dvac_rank_reward.py \
  tests/unit_tests/test_prism_dvac_rank_rloo.py
python -m py_compile \
  rlinf/algorithms/advantages.py \
  rlinf/algorithms/dvac_rank_reward.py \
  rlinf/config.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py
pytest -q \
  tests/unit_tests/test_prism_dvac_rank_rloo.py \
  tests/unit_tests/test_dvac_train_weighting.py

python examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  --cfg job --resolve \
  > /tmp/sz-prism-control-resolved-20260826.yaml
python examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  algorithm.adv_type=prism_rloo \
  algorithm.filter_rewards=false \
  algorithm.prism_dvac.enabled=true \
  --cfg job --resolve \
  > /tmp/sz-prism-method-resolved-20260826.yaml

grep -q '^  adv_type: grpo$' /tmp/sz-prism-control-resolved-20260826.yaml
grep -q '^    enabled: false$' /tmp/sz-prism-control-resolved-20260826.yaml
grep -q '^  adv_type: prism_rloo$' /tmp/sz-prism-method-resolved-20260826.yaml
grep -q '^  filter_rewards: false$' /tmp/sz-prism-method-resolved-20260826.yaml
grep -q '^    enabled: true$' /tmp/sz-prism-method-resolved-20260826.yaml

git status --short
git diff --stat
echo SZ_PRISM_MINIMAL_CHECKS_OK
