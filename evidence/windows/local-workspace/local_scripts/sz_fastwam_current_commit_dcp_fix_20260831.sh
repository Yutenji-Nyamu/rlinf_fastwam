#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
SRC=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/robotwin_move_stapler_pad_grpo_fastwam.dcp.yaml
EXPECTED=f730aff3ab5e11a584c5c8afa74e90f5fb8ffebd
TARGET=examples/embodiment/config/robotwin_move_stapler_pad_grpo_fastwam.yaml

test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$SRC"
install -m 0644 "$SRC" "$WT/$TARGET"
git -C "$WT" diff --check
test "$(git -C "$WT" diff --numstat -- "$TARGET")" = $'1\t1\texamples/embodiment/config/robotwin_move_stapler_pad_grpo_fastwam.yaml'
grep -F 'checkpoint_format: dcp' "$WT/$TARGET" >/dev/null
git -C "$WT" add "$TARGET"
git -C "$WT" commit -m 'fix(embodiment): use DCP for Fast-WAM FSDP2'
git -C "$WT" push personal codex/sz-fastwam-current-rlinf-grpo
head=$(git -C "$WT" rev-parse HEAD)
remote=$(git -C "$WT" ls-remote personal refs/heads/codex/sz-fastwam-current-rlinf-grpo | awk '{print $1}')
test "$head" = "$remote"
test -z "$(git -C "$WT" status --porcelain)"
printf 'HEAD=%s\nREMOTE=%s\nFASTWAM_DCP_FIX_PUSHED\n' "$head" "$remote"
