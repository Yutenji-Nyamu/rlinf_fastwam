set -euo pipefail
root=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
log=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab.download.log
pidfile=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab.download.pid
mkdir -p /data/chenyiteng/models/lerobot "$root"
if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "ALREADY_RUNNING pid=$(cat "$pidfile")"
  exit 0
fi
nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_ENABLE_HF_TRANSFER=0 \
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='SidneyXie/pi05_robotwin', revision='e49e2ab6c11f07511573b67261bd129e88d0a416', local_dir='$root', max_workers=4))" \
  >"$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidfile"
echo "STARTED pid=$pid root=$root log=$log"
sleep 2
ps -p "$pid" -o pid=,etime=,stat=,cmd=
tail -20 "$log" || true
