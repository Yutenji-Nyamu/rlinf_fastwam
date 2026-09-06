set -euo pipefail
base=/data/chenyiteng/projects/lerobot-sidney
src=$base/lerobot-0b067df57d21
venv=/home/chenyiteng/venvs/lerobot-v043-sidney-py310
log=$base/install-lerobot-v043-sidney-py310.log
pidfile=$base/install-lerobot-v043-sidney-py310.pid
mkdir -p "$base"
if [ ! -d "$src/.git" ]; then
  git clone -q https://github.com/huggingface/lerobot.git "$src"
fi
git -C "$src" fetch -q --tags origin
git -C "$src" checkout -q --detach 0b067df57d21d3a02d6c511f1609172fa39ac29b
test -z "$(git -C "$src" status --porcelain)"
if [ ! -x "$venv/bin/python" ]; then
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python -m venv --system-site-packages "$venv"
fi
printf '%s\n' 'numpy<2' 'torch==2.7.1' 'torchvision==0.22.1' > "$base/constraints-v043-py310.txt"
if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "ALREADY_RUNNING pid=$(cat "$pidfile")"
  exit 0
fi
nohup env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$venv/bin/python" -m pip install -e "$src[pi]" -c "$base/constraints-v043-py310.txt" \
  >"$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidfile"
echo "STARTED pid=$pid src=$(git -C "$src" rev-parse HEAD) venv=$venv log=$log"
sleep 2
tail -30 "$log" || true
echo '=== model download ==='
model=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
du -sh "$model" 2>/dev/null || true
find "$model" -maxdepth 2 -type f -printf '%s %p\n' 2>/dev/null | sort -nr | head -10 || true
if [ -f /data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab.download.pid ]; then
  dpid=$(cat /data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab.download.pid)
  ps -p "$dpid" -o pid=,etime=,stat=,cmd= || true
fi
