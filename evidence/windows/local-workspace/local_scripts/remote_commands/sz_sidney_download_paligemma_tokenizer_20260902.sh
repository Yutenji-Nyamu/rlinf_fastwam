set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/data/chenyiteng/cache/huggingface-sidney
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
"$venv/bin/python" - <<'PY'
from huggingface_hub import HfApi, snapshot_download
repo='google/paligemma-3b-pt-224'
api=HfApi(endpoint='https://hf-mirror.com')
info=api.model_info(repo)
print('TOKENIZER_REVISION', info.sha)
p=snapshot_download(
 repo_id=repo,
 revision=info.sha,
 allow_patterns=['tokenizer*','special_tokens_map.json','added_tokens.json','config.json'],
 cache_dir='/data/chenyiteng/cache/huggingface-sidney',
 endpoint='https://hf-mirror.com',
)
print('TOKENIZER_SNAPSHOT',p)
PY
