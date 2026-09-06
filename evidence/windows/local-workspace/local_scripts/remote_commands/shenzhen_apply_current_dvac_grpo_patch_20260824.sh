#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
HEAD=bfb99cce722015fe55bb3393bafb6f837e4cfa90

printf 'MARKER=SZ_APPLY_CURRENT_DVAC_GRPO_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" branch --show-current)" = codex/sz-current-pi0-dvac-grpo
test -z "$(git -C "$WT" status --short)"
git -C "$WT" apply --index --whitespace=error-all -
git -C "$WT" diff --cached --check
git -C "$WT" status --short
git -C "$WT" diff --cached --stat
printf 'MARKER=SZ_APPLY_CURRENT_DVAC_GRPO_OK\n'
