set -eu
dep=/root/autodl-tmp/RLinf/.venv/lib/python3.11/site-packages/openpi/models_pytorch
grep -R -n \
  'embed_tokens\|lm_head\|tie_word\|tie_weights' \
  "$dep/gemma_pytorch.py" \
  "$dep/pi0_pytorch.py" |
  head -120
sed -n '1,190p' "$dep/gemma_pytorch.py"
