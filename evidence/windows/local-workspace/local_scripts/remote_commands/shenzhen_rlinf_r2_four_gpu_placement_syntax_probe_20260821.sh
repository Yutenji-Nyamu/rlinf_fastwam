#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839

source "$VENV/bin/activate"
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROOT/examples/embodiment"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

EVAL_PLACEMENT='cluster.component_placement={env\,\ rollout:0-3}'
PPO_PLACEMENT='cluster.component_placement={actor\,\ env\,\ rollout:0-3}'

"$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" \
  --config-path "$ROOT/evaluations/robotwin" \
  --config-name robotwin_adjust_bottle_openpi_eval \
  "$EVAL_PLACEMENT" --cfg job --resolve \
  | sed -n '/^cluster:/,/^runner:/p'

"$VENV/bin/python" "$ROOT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_ppo_openpi \
  "$PPO_PLACEMENT" --cfg job --resolve \
  | sed -n '/^cluster:/,/^runner:/p'

printf '%s\n' 'R2_FOUR_GPU_PLACEMENT_SYNTAX_PROBE_OK'
