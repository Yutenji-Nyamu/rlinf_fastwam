#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
branch=codex/idea2-dvac-residual-downweight

git -C "$repo" status --short --branch
git -C "$repo" push -u personal "$branch"
git -C "$repo" status --short --branch
git -C "$repo" rev-parse HEAD
