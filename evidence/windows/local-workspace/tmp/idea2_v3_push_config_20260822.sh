#!/usr/bin/env bash
set -euo pipefail

source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
expected=eb2a09176c362c7386895ca4f3680b92aeb0ee5b

test "$(git -C "$source_root" rev-parse HEAD)" = "$expected"
printf 'PRE_PUSH\n'
git -C "$source_root" status --short
git -C "$source_root" remote get-url personal

timeout 45 git -C "$source_root" push personal HEAD:codex/idea2-dvac-residual-downweight

printf 'POST_PUSH\n'
git -C "$source_root" rev-parse HEAD
git -C "$source_root" ls-remote personal refs/heads/codex/idea2-dvac-residual-downweight
git -C "$source_root" status --short
