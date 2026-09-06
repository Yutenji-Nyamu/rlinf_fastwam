set -euo pipefail
source /etc/profile.d/mihomo-proxy.sh 2>/dev/null || true
basepy=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
venv=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
if [ ! -x "$venv/bin/python" ]; then
  "$basepy" -m venv --system-site-packages "$venv"
fi
"$venv/bin/python" -m pip install -q --upgrade 'pip<26'
"$venv/bin/python" -m pip install 'transformers>=5.4,<5.6' 'huggingface-hub>=1,<2' 'draccus==0.10.0' 'gymnasium>=1.1.1,<2'
"$venv/bin/python" -m pip install --no-deps -e "$src"
"$venv/bin/python" - <<'PY'
import sys, importlib.metadata as m
print(sys.version)
for x in ['lerobot','torch','torchvision','numpy','transformers','huggingface-hub','draccus','gymnasium','safetensors']:
 print(x, m.version(x))
PY
