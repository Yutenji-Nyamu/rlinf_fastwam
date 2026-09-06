#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
overlay=$worktree/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250.yaml
seeds=$worktree/rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json
expected_old_head=bdd875283b3f3516c439e5c79c902cf5c2da58b6
branch=codex/sz-rlt-pi0-robotwin-ar

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_old_head"
test "$(git -C "$worktree" branch --show-current)" = "$branch"
test -s "$overlay"
test -s "$seeds"
status=$(git -C "$worktree" status --porcelain)
test "$(printf '%s\n' "$status" | sed '/^$/d' | wc -l)" -eq 2
test -z "$(printf '%s\n' "$status" | grep -v -E '^\?\? (examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250.yaml|rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json)$' || true)"

git -C "$worktree" add \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250.yaml \
  rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json
git -C "$worktree" diff --cached --check
git -C "$worktree" commit -m 'feat(rlt): add current RoboTwin formal protocol'
git -C "$worktree" push personal "HEAD:$branch"

printf 'new_head='; git -C "$worktree" rev-parse HEAD
printf 'remote_head='; git -C "$worktree" rev-parse "personal/$branch"
printf 'overlay_sha256='; sha256sum "$overlay" | awk '{print $1}'
printf 'seeds_sha256='; sha256sum "$seeds" | awk '{print $1}'
printf 'dirty='; git -C "$worktree" status --porcelain | wc -l
git -C "$worktree" show --stat --oneline --decorate HEAD
