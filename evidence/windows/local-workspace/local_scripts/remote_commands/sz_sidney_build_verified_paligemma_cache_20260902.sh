set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
root=/data/chenyiteng/models/lerobot/paligemma-tokenizer-verified
mirror=$root/leo009-39996beb6fb17c5d16a50d3ef8f7a96ad9d03986
official_rev=35e4f46485b4d07967e7e9935bc3786aad50687c
snapshot=/data/chenyiteng/cache/huggingface-sidney/hub/models--google--paligemma-3b-pt-224/snapshots/$official_rev
mkdir -p "$root" "$snapshot" "$(dirname "$(dirname "$snapshot")")/refs"
"$venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
p=snapshot_download(
 repo_id='leo009/paligemma-3b-pt-224',
 revision='39996beb6fb17c5d16a50d3ef8f7a96ad9d03986',
 allow_patterns=['added_tokens.json','config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json'],
 local_dir='/data/chenyiteng/models/lerobot/paligemma-tokenizer-verified/leo009-39996beb6fb17c5d16a50d3ef8f7a96ad9d03986',
 endpoint='https://hf-mirror.com',
)
print(p)
PY
for f in added_tokens.json config.json special_tokens_map.json tokenizer.json tokenizer_config.json; do
  cp -f "$mirror/$f" "$snapshot/$f"
done
cp -f /home/chenyiteng/.cache/openpi/big_vision/paligemma_tokenizer.model "$snapshot/tokenizer.model"
printf '%s' "$official_rev" > "$(dirname "$(dirname "$snapshot")")/refs/main"
echo '=== hashes ==='
for f in added_tokens.json config.json special_tokens_map.json tokenizer_config.json; do
  printf '%s ' "$f"; git hash-object "$snapshot/$f"
done
sha256sum "$snapshot/tokenizer.json" "$snapshot/tokenizer.model"
echo "CACHE_SNAPSHOT=$snapshot"
