set -euo pipefail
pid=1353070
if kill -0 "$pid" 2>/dev/null; then
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in *lerobot-v043-sidney-py310/bin/python*pip*install*) echo "STOP_OWNED_INSTALL $pid"; kill -TERM "$pid";; *) echo "REFUSE $pid $cmd"; exit 2;; esac
fi
sleep 2
source /etc/profile.d/mihomo-proxy.sh
base=/data/chenyiteng/projects/lerobot-sidney
venv=/home/chenyiteng/venvs/lerobot-v043-sidney-py310
src=$base/lerobot-0b067df57d21
tsrc=$base/transformers-dcddb970176382c0fcf4521b0c0e6fc15894dfe0
"$venv/bin/python" -m pip install \
  --no-deps -e "$tsrc" -e "$src"
"$venv/bin/python" -m pip install \
  'draccus==0.10.0' \
  'huggingface-hub>=0.34.2,<0.36.0' \
  'scipy>=1.10.1,<1.15' \
  'gymnasium>=1.1.1,<2.0.0' \
  'opencv-python-headless>=4.9.0,<4.13.0' \
  'termcolor>=2.4.0,<4.0.0' \
  -c "$base/constraints-v043-py310.txt"
echo '=== imports ==='
"$venv/bin/python" - <<'PY'
import importlib.metadata as m
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies import make_pre_post_processors
for x in ['lerobot','transformers','huggingface-hub','draccus','torch','torchvision','gymnasium','scipy']:
 try: print(x,m.version(x))
 except Exception as e: print(x,'ERR',e)
print('PI05Policy', PI05Policy)
print('processors', make_pre_post_processors)
PY
