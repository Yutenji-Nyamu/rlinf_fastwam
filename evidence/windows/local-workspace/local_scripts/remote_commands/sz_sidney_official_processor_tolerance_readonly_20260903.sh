set -eu
ROOT=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
grep -R -n -C 3 -E 'assert_close|allclose|atol|rtol' \
  "$ROOT/tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py" \
  "$ROOT/tests/policies/pi0_pi05/utils/openpi_parity.py" | head -240
