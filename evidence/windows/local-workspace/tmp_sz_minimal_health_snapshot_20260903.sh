set -u
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
echo "NOW_CST=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
echo '===GPU===' 
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '===GPU_PROCESSES===' 
mapfile -t pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
if test "${#pids[@]}" -eq 0; then echo NONE; else
  for pid in "${pids[@]}"; do ps -p "$pid" -o user=,pid=,etime=,rss=,comm=,args= | cut -c1-300 || true; done
fi
echo '===MEM===' 
awk '/MemTotal:|MemAvailable:|SwapTotal:|SwapFree:/ {print}' /proc/meminfo
cat /proc/pressure/memory
echo '===DISK===' 
df -hT / /home /data
echo '===RAY===' 
ps -p 321933 -o user=,pid=,stat=,etime=,rss=,comm= --no-headers || true
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status | sed -n '1,24p' || true
echo '===NETWORK===' 
for url in https://github.com https://huggingface.co; do
  printf '%s ' "$url"
  HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
    curl -L -o /dev/null -sS --connect-timeout 8 --max-time 15 \
      -w 'code=%{http_code} total=%{time_total}s\n' "$url" || echo FAILED
done
