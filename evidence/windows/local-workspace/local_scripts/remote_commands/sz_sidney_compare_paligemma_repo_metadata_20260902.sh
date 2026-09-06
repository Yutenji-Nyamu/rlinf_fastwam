set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
"$venv/bin/python" - <<'PY'
from huggingface_hub import HfApi
api=HfApi(endpoint='https://hf-mirror.com')
names={'added_tokens.json','config.json','special_tokens_map.json','tokenizer.json','tokenizer.model','tokenizer_config.json'}
for repo in ['google/paligemma-3b-pt-224','leo009/paligemma-3b-pt-224']:
 info=api.model_info(repo, files_metadata=True)
 print('REPO',repo,'REV',info.sha)
 for s in info.siblings:
  if s.rfilename in names:
   print(s.rfilename,'size',s.size,'blob',getattr(s,'blob_id',None),'lfs',getattr(s.lfs,'sha256',None) if s.lfs else None)
PY
