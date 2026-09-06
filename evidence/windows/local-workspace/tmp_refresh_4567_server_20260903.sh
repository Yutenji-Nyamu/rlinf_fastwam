set -euo pipefail

echo 'TIME'
date -Is

echo 'GPU'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

echo 'GPU_PROCESSES'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | while IFS=, read -r uuid pid mem; do
  pid="${pid// /}"
  user=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
  cmd=$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-100 || true)
  printf '%s pid=%s mem=%sMiB user=%s cmd=%s\n' "$uuid" "$pid" "${mem// /}" "$user" "$cmd"
done

echo 'MEM_LOAD'
free -h | sed -n '1,2p'
uptime
cat /proc/pressure/memory | head -n 1

echo 'DISK'
df -h / /home /data | awk 'NR==1 || !seen[$6]++'

echo 'NETWORK'
systemctl is-active mihomo 2>/dev/null || true
for url in https://github.com https://huggingface.co; do
  code=$(HTTPS_PROXY=http://127.0.0.1:7890 curl -L -o /dev/null -sS --max-time 8 -w '%{http_code}' "$url" || true)
  printf '%s %s\n' "$url" "$code"
done

echo 'SIDNEY_STATUS'
SIDNEY=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney-grpo/runs
find "$SIDNEY" -maxdepth 4 -type f \( -name '*.log' -o -name 'driver.stdout' \) -mmin -180 2>/dev/null | head -n 10
ps -p 2660157 -o pid=,etime=,stat=,args= 2>/dev/null || true

echo 'FASTWAM_STATUS'
FAST=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2
ps -p 2812508 -o pid=,etime=,stat=,args= 2>/dev/null || true
find "$FAST" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 12
grep -R -E 'Global Step|success_rate|eval.*success|OIDN|pthread_key_create|PyGILState|CUDA out of memory|Traceback|WorkerCrashed|exit_code' "$FAST" --include='*.log' --include='*.out' --include='*.txt' 2>/dev/null | tail -n 80 || true
