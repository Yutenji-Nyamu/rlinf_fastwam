set -euo pipefail
find /data/chenyiteng/models /data/chenyiteng/cache /home/chenyiteng/.cache -maxdepth 9 \
  \( -iname '*paligemma*' -o -iname 'tokenizer.json' -o -iname 'tokenizer.model' -o -iname 'tokenizer_config.json' \) \
  -printf '%y %s %p\n' 2>/dev/null | head -240 || true
