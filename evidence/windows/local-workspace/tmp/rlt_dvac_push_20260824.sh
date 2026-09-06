#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
branch=codex/rlt-teacher-dvac-weighting

git -C "$repo" push --set-upstream personal "$branch"
local_sha=$(git -C "$repo" rev-parse HEAD)
remote_sha=$(git -C "$repo" ls-remote --heads personal "refs/heads/$branch" | awk '{print $1}')
printf 'LOCAL_SHA=%s\nREMOTE_SHA=%s\n' "$local_sha" "$remote_sha"
test "$local_sha" = "$remote_sha"
git -C "$repo" status --short --branch
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
