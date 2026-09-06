#!/usr/bin/env bash
set -euo pipefail
echo '=== lerobot roots ==='
find /data/chenyiteng/projects -maxdepth 3 -type d -iname '*lerobot*' | head -20
echo '=== existing sidney eval/parity scripts ==='
find /data/chenyiteng/projects /data/chenyiteng/runs -type f \( -iname '*sidney*.py' -o -iname '*parity*.py' -o -iname '*oracle*.py' \) 2>/dev/null | head -60
ROOT=/data/chenyiteng/projects/lerobot-sidney
if [ -d "$ROOT" ]; then
  rg -n "predict_action_chunk|select_action|noise|sample_actions|unnormal|preprocess|postprocess" "$ROOT/src/lerobot/policies/pi05" "$ROOT/tests/policies/pi0_pi05" 2>/dev/null | head -180 || true
fi
