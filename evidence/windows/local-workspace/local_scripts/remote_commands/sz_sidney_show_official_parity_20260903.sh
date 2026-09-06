#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
find "$ROOT/tests/policies/pi0_pi05" -maxdepth 2 -type f -printf '%P\n' | sort
echo '=== parity util ==='
sed -n '1,300p' "$ROOT/tests/policies/pi0_pi05/utils/openpi_parity.py"
echo '=== pi05 tests ==='
rg -l "openpi_parity|Pi05|PI05" "$ROOT/tests/policies/pi0_pi05" | while read -r f; do echo "--- $f"; sed -n '1,260p' "$f"; done
echo '=== model sampling signatures ==='
rg -n "def (select_action|predict_action_chunk|sample_actions|forward)" "$ROOT/src/lerobot/policies/pi05" -g '*.py'
