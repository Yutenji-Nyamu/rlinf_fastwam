#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
EXPECTED_HEAD=d7c3ca7e2ddfc8d0b3c376ec6d30ba89b965a5dc

test "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$ROOT" status --porcelain)"
test "$(git -C "$ROOT" rev-list --left-right --count HEAD...@{upstream})" = $'1\t0'
timeout 240s git -C "$ROOT" push personal codex/rlt-pi0-robotwin
test "$(git -C "$ROOT" rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
git -C "$ROOT" ls-remote personal refs/heads/codex/rlt-pi0-robotwin
