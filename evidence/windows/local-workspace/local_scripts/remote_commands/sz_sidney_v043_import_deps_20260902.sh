set -euo pipefail
source /etc/profile.d/mihomo-proxy.sh
venv=/home/chenyiteng/venvs/lerobot-v043-sidney-py310
"$venv/bin/python" -m pip install 'diffusers>=0.27.2,<0.36.0' 'pyserial>=3.5,<4.0' 'deepdiff>=7.0.1,<9.0.0' 'pynput>=1.7.7,<1.9.0'
"$venv/bin/python" - <<'PY'
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
print('PI05_IMPORT_OK', PI05Policy.__name__, make_pre_post_processors.__name__)
PY
