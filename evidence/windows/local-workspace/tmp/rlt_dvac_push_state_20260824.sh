#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short --branch
git -C "$repo" rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || echo NO_UPSTREAM
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD' 2>/dev/null || true
pgrep -af 'git.*rlt-teacher-dvac' || true
