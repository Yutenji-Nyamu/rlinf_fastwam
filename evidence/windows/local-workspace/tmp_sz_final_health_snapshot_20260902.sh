set -u
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
echo "NOW_CST=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
echo '===GPU===' 
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '===GPU_COMPUTE_OWNERS===' 
mapfile -t pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
if test "${#pids[@]}" -eq 0; then
  echo NONE
else
  for pid in "${pids[@]}"; do
    ps -p "$pid" -o user=,pid=,etime=,rss=,stat=,comm=,args= | cut -c1-320 || true
  done
fi
echo '===RAM===' 
free -h
cat /proc/pressure/memory
echo '===DISK===' 
df -hT / /home /data
echo '===SHARED_RAY===' 
ps -p 321933 -o user=,pid=,stat=,etime=,rss=,args= | cut -c1-500 || true
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status | sed -n '1,28p' || true
echo '===NETWORK===' 
ps -eo user=,pid=,stat=,etime=,comm=,args= | grep -E '[m]ihomo' | cut -c1-320 || true
for url in https://github.com https://huggingface.co; do
  printf '%s ' "$url"
  HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
    curl -L -o /dev/null -sS --connect-timeout 8 --max-time 15 \
      -w 'code=%{http_code} connect=%{time_connect}s total=%{time_total}s remote=%{remote_ip}\n' "$url" || echo CURL_FAILED
done
echo '===USER_ACTIVITY_TOP_RSS===' 
users=$(getent passwd | awk -F: '$3>=1000 && $3<65534 {print $1}' | paste -sd'|' -)
ps -eo user=,pid=,etime=,rss=,stat=,comm=,args= --sort=-rss | \
  awk -v re="^(${users})$" '$1 ~ re {print}' | head -n 40 | cut -c1-360
echo '===USER_PROCESS_COUNTS===' 
ps -eo user=,comm= --no-headers | awk -v re="^(${users})$" '$1 ~ re {k=$1" "$2;c[k]++} END{for(k in c) print c[k],k}' | sort -k2,2 -k1,1nr | head -n 100
