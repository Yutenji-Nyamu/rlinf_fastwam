set -euo pipefail
echo '=== identity/time ==='
date -Is
id
echo '=== gpu4-5 ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '5,6p'
echo '=== candidate proxy/profile ==='
env | grep -Ei '^(http|https|all|no)_proxy=|^HF_' || true
find /etc/profile.d /home/chenyiteng -maxdepth 2 -type f \( -iname '*proxy*' -o -iname '*mihomo*' -o -iname '*clash*' -o -iname '*hf*' \) -printf '%p\n' 2>/dev/null | head -80
echo '=== candidate sources/env ==='
for p in \
  /data/chenyiteng/projects/lerobot \
  /data/chenyiteng/projects/LeRobot \
  /data/chenyiteng/projects/RoboTwin \
  /home/chenyiteng/RoboTwin \
  /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711; do
  if [ -e "$p" ]; then
    echo "FOUND $p"
    if [ -d "$p/.git" ]; then git -C "$p" rev-parse HEAD; git -C "$p" status --short | head; fi
  fi
done
for p in \
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128 \
  /home/chenyiteng/miniforge3/envs/RoboTwin; do
  if [ -x "$p/bin/python" ]; then
    echo "PY $p"
    "$p/bin/python" - <<'PY'
import sys, importlib.util
print(sys.version)
for x in ['torch','lerobot','sapien','robotwin','gymnasium','transformers','huggingface_hub']:
 print(x, bool(importlib.util.find_spec(x)))
try:
 import torch; print('torch',torch.__version__)
except Exception as e: print('torcherr',type(e).__name__,str(e))
PY
  fi
done
echo '=== git remote revision ==='
git ls-remote https://github.com/huggingface/lerobot.git refs/heads/main refs/tags/v0.4.3 refs/tags/v0.5.0 refs/tags/v0.6.0 || true
echo '=== hf api direct/mirror ==='
curl -LIsS --max-time 20 https://huggingface.co/api/models/SidneyXie/pi05_robotwin | head -5 || true
curl -LIsS --max-time 20 https://hf-mirror.com/api/models/SidneyXie/pi05_robotwin | head -5 || true
echo '=== disk ==='
df -h /data /home
