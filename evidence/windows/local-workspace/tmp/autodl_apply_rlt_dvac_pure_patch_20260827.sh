#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
archive=/tmp/rlt_dvac_pure_patch_20260827.tar

test "$(git -C "$repo" rev-parse HEAD)" = 848b61278687702ea717c56b3734f1486cea3b95
test -z "$(git -C "$repo" status --short)"
tar -xf "$archive" -C "$repo"
git -C "$repo" diff --check
git -C "$repo" status --short
git -C "$repo" diff --stat
