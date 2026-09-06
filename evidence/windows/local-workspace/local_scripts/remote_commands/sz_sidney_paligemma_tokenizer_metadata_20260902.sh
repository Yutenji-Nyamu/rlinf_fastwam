set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
"$venv/bin/python" - <<'PY'
from huggingface_hub import HfApi
api=HfApi(endpoint='https://hf-mirror.com')
info=api.model_info('google/paligemma-3b-pt-224', files_metadata=True)
print('REV',info.sha)
for s in info.siblings:
 if 'token' in s.rfilename or s.rfilename in ('config.json','added_tokens.json','special_tokens_map.json'):
  print(s.rfilename, s.size, s.lfs)
PY
sha256sum /home/chenyiteng/.cache/openpi/big_vision/paligemma_tokenizer.model
cat /home/chenyiteng/.cache/openpi/.cache/huggingface/download/big_vision/paligemma_tokenizer.model.metadata 2>/dev/null || true
