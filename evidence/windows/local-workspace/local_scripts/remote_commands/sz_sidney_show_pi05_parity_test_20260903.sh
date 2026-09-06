#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
sed -n '1,360p' "$ROOT/tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py"
echo '=== util tail ==='
sed -n '250,520p' "$ROOT/tests/policies/pi0_pi05/utils/openpi_parity.py"
echo '=== pi05 modeling action methods ==='
grep -R -nE "def (select_action|predict_action_chunk|sample_actions|prepare_state|_preprocess_images)" "$ROOT/src/lerobot/policies/pi05" | head -80
