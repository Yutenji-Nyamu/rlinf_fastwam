#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
FILE=examples/embodiment/config/robotwin_move_stapler_pad_grpo_fastwam.yaml

test "$(git -C "$WT" rev-parse HEAD)" = 81076b13e91a3db0b8def6b48dd6e1db2897cb2b
test "$(git -C "$WT" status --short)" = " M $FILE"
git -C "$WT" diff --check
git -C "$WT" add -- "$FILE"
git -C "$WT" diff --cached --quiet && exit 31
git -C "$WT" commit -m 'fix(embodiment): offload RoboTwin envs for Fast-WAM'
git -C "$WT" push personal codex/sz-fastwam-current-rlinf-grpo
test -z "$(git -C "$WT" status --porcelain)"
head=$(git -C "$WT" rev-parse HEAD)
remote=$(git -C "$WT" ls-remote personal refs/heads/codex/sz-fastwam-current-rlinf-grpo | awk '{print $1}')
test "$head" = "$remote"
printf 'HEAD=%s\nREMOTE=%s\nFASTWAM_ENV_OFFLOAD_FIX_PUSHED\n' "$head" "$remote"
