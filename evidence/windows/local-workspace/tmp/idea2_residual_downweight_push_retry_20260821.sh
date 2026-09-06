#!/usr/bin/env bash
set -u

ROOT=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
BRANCH=codex/idea2-dvac-residual-downweight

git -C "$ROOT" rev-parse HEAD
git -C "$ROOT" status --short --branch
git -C "$ROOT" remote -v
GIT_TERMINAL_PROMPT=0 timeout 60s git -C "$ROOT" push -u personal "$BRANCH"
rc=$?
echo PUSH_RC="$rc"
git -C "$ROOT" status --short --branch
exit "$rc"
