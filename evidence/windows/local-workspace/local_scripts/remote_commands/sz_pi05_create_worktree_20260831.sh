#!/usr/bin/env bash
set -euo pipefail

SOURCE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
BASE=74617ced87d64045ab6850d0efd90956a494af66
BRANCH=codex/sz-pi05-robotwin-rl

test "$(git -C "$SOURCE" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$SOURCE" status --porcelain)"
test ! -e "$TARGET"
test -z "$(git -C "$SOURCE" branch --list "$BRANCH")"
test -z "$(git -C "$SOURCE" ls-remote --heads personal "$BRANCH")"

git -C "$SOURCE" worktree add -b "$BRANCH" "$TARGET" "$BASE"
test "$(git -C "$TARGET" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$TARGET" status --porcelain)"
printf 'branch=%s\nhead=%s\nworktree=%s\nPI05_WORKTREE_CREATED\n' \
  "$(git -C "$TARGET" branch --show-current)" \
  "$(git -C "$TARGET" rev-parse HEAD)" \
  "$TARGET"
