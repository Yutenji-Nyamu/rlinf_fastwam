set -euo pipefail
unset HF_HOME || true
export HF_ENDPOINT=https://hf-mirror.com
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
"$venv/bin/python" - <<'PY'
from huggingface_hub import HfApi
try:
 info=HfApi(endpoint='https://hf-mirror.com').whoami()
 print('HF_AUTH_OK', info.get('name') or info.get('fullname') or 'authenticated')
except Exception as e:
 print('HF_AUTH_NONE',type(e).__name__,str(e).splitlines()[0])
PY
