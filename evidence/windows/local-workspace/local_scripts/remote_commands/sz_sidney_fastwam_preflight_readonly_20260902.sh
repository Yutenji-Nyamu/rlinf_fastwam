set -eu

echo '== identity/time =='
date '+%F %T %Z'
id

echo '== gpu =='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo '== memory/disk =='
free -h
df -h / /home /data

echo '== relevant envs and sources =='
for p in \
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128 \
  /home/chenyiteng/venvs/fasterwam-official \
  /home/chenyiteng/miniforge3/envs/RoboTwin \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin \
  /data/chenyiteng/projects/robotwin-native/RoboTwin \
  /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711@7faa71108368fbb3b6885649f112af607427a2d4; do
  if test -e "$p"; then
    echo "EXISTS $p"
  else
    echo "MISSING $p"
  fi
done

for py in \
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python \
  /home/chenyiteng/miniforge3/envs/RoboTwin/bin/python \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python; do
  if test -x "$py"; then
    "$py" -c 'import sys,torch; print(sys.executable, sys.version.split()[0], torch.__version__)' || true
  fi
done

echo '== fastwam current packet =='
packet=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2
if test -d "$packet"; then
  find "$packet" -maxdepth 2 -type f -printf '%P\t%s\n' | sort
  for f in "$packet"/*.yaml "$packet"/*.sh "$packet"/*.json; do
    test -f "$f" || continue
    echo "--- $f"
    sed -n '1,260p' "$f"
  done
fi

echo '== caches =='
find /data/chenyiteng /home/chenyiteng/cache -maxdepth 5 -type d \
  \( -iname '*paligemma*' -o -iname '*pi05_robotwin*' -o -iname '*lerobot*' \) \
  -printf '%p\n' 2>/dev/null | head -100

echo '== network bounded =='
timeout 15s curl -I -L -sS -o /dev/null -w 'github_direct=%{http_code}\n' https://github.com || true
timeout 15s curl -I -L -sS -o /dev/null -w 'hf_direct=%{http_code}\n' https://huggingface.co/SidneyXie/pi05_robotwin || true
if test -f /etc/profile.d/mihomo.sh; then
  . /etc/profile.d/mihomo.sh
elif test -f /etc/profile.d/proxy.sh; then
  . /etc/profile.d/proxy.sh
fi
timeout 15s curl -I -L -sS -o /dev/null -w 'hf_login_route=%{http_code}\n' https://huggingface.co/SidneyXie/pi05_robotwin || true
