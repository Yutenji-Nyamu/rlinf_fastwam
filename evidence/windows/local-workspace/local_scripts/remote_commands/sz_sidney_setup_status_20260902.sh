set -euo pipefail
echo '=== model ==='
model=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
du -sh "$model" 2>/dev/null || true
find "$model" -maxdepth 2 -type f -printf '%s %p\n' 2>/dev/null | sort -nr | head -12 || true
dpidfile=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab.download.pid
if [ -f "$dpidfile" ]; then ps -p "$(cat "$dpidfile")" -o pid=,etime=,stat=,cmd= || true; fi
tail -20 /data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab.download.log 2>/dev/null || true
echo '=== env install ==='
ipf=/data/chenyiteng/projects/lerobot-sidney/install-lerobot-v043-sidney-py310.pid
if [ -f "$ipf" ]; then ps -p "$(cat "$ipf")" -o pid=,etime=,stat=,cmd= || true; fi
tail -40 /data/chenyiteng/projects/lerobot-sidney/install-lerobot-v043-sidney-py310.log 2>/dev/null || true
echo '--- retry ---'
rpf=/data/chenyiteng/projects/lerobot-sidney/install-lerobot-v043-sidney-py310.retry.pid
if [ -f "$rpf" ]; then ps -p "$(cat "$rpf")" -o pid=,etime=,stat=,cmd= || true; fi
tail -50 /data/chenyiteng/projects/lerobot-sidney/install-lerobot-v043-sidney-py310.retry.log 2>/dev/null || true
echo '=== venv basic ==='
if [ -x /home/chenyiteng/venvs/lerobot-v043-sidney-py310/bin/python ]; then
/home/chenyiteng/venvs/lerobot-v043-sidney-py310/bin/python - <<'PY'
import sys, importlib.util
print(sys.version)
for x in ['lerobot','draccus','transformers','torch','safetensors']:
 print(x, bool(importlib.util.find_spec(x)))
PY
else
  echo 'venv not created yet'
fi
