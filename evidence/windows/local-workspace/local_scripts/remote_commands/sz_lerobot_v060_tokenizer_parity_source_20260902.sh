set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
grep -R -n -E 'PaligemmaTokenizer|paligemma_tokenizer|AutoTokenizer|tokenizer.model' \
  "$src/tests/policies/pi0_pi05" "$src/src/lerobot/policies/pi05" 2>/dev/null | head -120 || true
sed -n '1,190p' "$src/tests/policies/pi0_pi05/utils/openpi_parity.py" 2>/dev/null || true
