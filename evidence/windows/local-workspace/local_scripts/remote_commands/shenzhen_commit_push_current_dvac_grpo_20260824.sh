#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
BRANCH=codex/sz-current-pi0-dvac-grpo
PARENT=bfb99cce722015fe55bb3393bafb6f837e4cfa90

printf 'MARKER=SZ_COMMIT_PUSH_CURRENT_DVAC_GRPO_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
test "$(git -C "$WT" rev-parse HEAD)" = "$PARENT"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$WT" diff --name-only)"
git -C "$WT" diff --cached --check
mapfile -t names < <(git -C "$WT" diff --cached --name-only)
test "${#names[@]}" = 5

identity_ref=personal/main
export GIT_AUTHOR_NAME="$(git -C "$WT" show -s --format=%an "$identity_ref")"
export GIT_AUTHOR_EMAIL="$(git -C "$WT" show -s --format=%ae "$identity_ref")"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
remote_before=$(timeout 60s git -C "$WT" ls-remote --heads personal "refs/heads/$BRANCH")
test -z "$remote_before"
git -C "$WT" commit \
  -m "feat(embodiment): add current pi0 DVAC GRPO weighting" \
  -m "Port the proven global-z [0,2] rule onto current typed trajectories and add exact recent-stat resume."
commit=$(git -C "$WT" rev-parse HEAD)
test "$(git -C "$WT" rev-parse HEAD^)" = "$PARENT"
test -z "$(git -C "$WT" status --short)"
git -C "$WT" show --stat --oneline --decorate --no-renames "$commit"
timeout 90s git -C "$WT" push --set-upstream personal "HEAD:refs/heads/$BRANCH"
remote_after=$(timeout 60s git -C "$WT" ls-remote --heads personal "refs/heads/$BRANCH" | awk '{print $1}')
test "$remote_after" = "$commit"
test -z "$(git -C "$WT" status --short)"
printf 'commit=%s\nremote=%s\n' "$commit" "$remote_after"
printf 'MARKER=SZ_COMMIT_PUSH_CURRENT_DVAC_GRPO_OK\n'
