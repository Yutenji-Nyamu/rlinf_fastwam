#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711

printf '%s\n' '=== INFER_ACTION_TAIL ==='
sed -n '1080,1235p' "$WT/src/fastwam/models/wan22/fastwam.py"
printf '%s\n' '=== MODEL_TRACE_PREVIOUS_COMMIT ==='
git -C "$WT" show HEAD:src/fastwam/models/wan22/fastwam.py | sed -n '1080,1235p'
printf '%s\n' '=== ROBOTWIN_LOCATIONS ==='
find /data/chenyiteng/projects /data/chenyiteng/official -maxdepth 6 -type f -name eval_policy.py -print 2>/dev/null || true
find /data/chenyiteng/projects /data/chenyiteng/official -maxdepth 7 -type f -path '*/task_config/demo_clean.yml' -print 2>/dev/null || true
printf '%s\n' '=== THIRD_PARTY_PATHS ==='
ls -la "$WT/third_party" 2>/dev/null || true
find "$WT" -maxdepth 4 -type l -print -exec readlink -f {} \; 2>/dev/null || true
