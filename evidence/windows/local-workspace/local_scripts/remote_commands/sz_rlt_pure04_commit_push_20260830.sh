#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421
branch=codex/sz-rlt-dvac-pure-single-gpu
base=8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1

test "$(git -C "$worktree" rev-parse HEAD)" = "$base"
test "$(git -C "$worktree" branch --show-current)" = "$branch"
git -C "$worktree" diff --check

paths=(
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control.yaml
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_dvac_pure04.yaml
  rlinf/algorithms/rlt/dvac_weighting.py
  rlinf/algorithms/rlt/rollout.py
  rlinf/algorithms/rlt/transition.py
  rlinf/models/embodiment/openpi/openpi_action_model.py
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
  tests/unit_tests/test_rlt_dvac_weighting.py
  tests/unit_tests/test_robotwin_rlt_current_port.py
)

git -C "$worktree" add -- "${paths[@]}"
git -C "$worktree" diff --cached --check
test "$(git -C "$worktree" diff --cached --name-only | wc -l)" = 9
git -C "$worktree" commit -m 'feat(rlt): port single-GPU Pure04 to current runtime'
git -C "$worktree" push -u personal "HEAD:refs/heads/$branch"

head=$(git -C "$worktree" rev-parse HEAD)
test "$head" = "$(git -C "$worktree" rev-parse "personal/$branch")"
test -z "$(git -C "$worktree" status --porcelain)"
printf 'branch=%s\nhead=%s\nremote_head=%s\nCOMMIT_PUSH_OK\n' \
  "$branch" "$head" "$(git -C "$worktree" rev-parse "personal/$branch")"
