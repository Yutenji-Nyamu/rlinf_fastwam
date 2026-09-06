#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
echo '[HEAD]'
git rev-parse HEAD
echo '[STATUS]'
git status --short
echo '[REMOTES]'
git remote -v
echo '[BRANCH]'
git branch -vv --no-abbrev | sed -n '/^\*/p'
