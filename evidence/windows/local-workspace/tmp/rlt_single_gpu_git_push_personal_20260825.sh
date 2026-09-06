#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
git push personal codex/rlt-teacher-dvac-weighting
git rev-parse HEAD
git status --short
