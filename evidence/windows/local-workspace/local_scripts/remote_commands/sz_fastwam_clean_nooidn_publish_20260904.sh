#!/usr/bin/env bash
set -euo pipefail
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-clean-oidn-off-20260904
BRANCH=codex/sz-robotwin-clean-oidn-off
test "$(git -C "$RT" rev-parse HEAD)" = f3e30a83365cdda1165911422dc3ce73e703201e
test -z "$(git -C "$RT" status --porcelain)"
test "$(git -C "$RT" remote get-url personal)" = https://github.com/Yutenji-Nyamu/RoboTwin.git
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
timeout 45s git -C "$RT" push personal "$BRANCH:$BRANCH"
test "$(git -C "$RT" rev-parse HEAD)" = "$(git -C "$RT" rev-parse "personal/$BRANCH")"
git -C "$RT" status --short --branch
date -Is
