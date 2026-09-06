#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
CONFIG=examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
EXPECTED_HEAD=66dc388e464660f0ed6a8c48b7a188731d3dbbbe
EXPECTED_SHA=c293bc476ec7458c6bfc5c5c59393e48b286f3e12007f3039ccc282e30645a4c

test "$(git -C "$RLT_ROOT" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$RLT_ROOT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$RLT_ROOT" rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
test "$(sha256sum "$RLT_ROOT/$CONFIG" | awk '{print $1}')" = "$EXPECTED_SHA"
test "$(git -C "$RLT_ROOT" status --porcelain | wc -l)" -eq 1
test "$(git -C "$RLT_ROOT" status --porcelain | sed 's/^...//')" = "$CONFIG"
git -C "$RLT_ROOT" diff --check
git -C "$RLT_ROOT" add -- "$CONFIG"
test "$(git -C "$RLT_ROOT" diff --cached --name-only)" = "$CONFIG"
git -C "$RLT_ROOT" diff --cached --check
git -C "$RLT_ROOT" commit -m "chore(rlt): bind Stage 1 to clean50 dataset"
git -C "$RLT_ROOT" push personal codex/rlt-pi0-robotwin
test -z "$(git -C "$RLT_ROOT" status --porcelain)"
test "$(git -C "$RLT_ROOT" rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
printf 'HEAD\t%s\n' "$(git -C "$RLT_ROOT" rev-parse HEAD)"
git -C "$RLT_ROOT" ls-remote personal refs/heads/codex/rlt-pi0-robotwin
