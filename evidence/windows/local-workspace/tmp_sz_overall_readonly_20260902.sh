set -u

echo '=== IDENTITY_TIME ==='
date --iso-8601=seconds
hostname
id
uptime
printf 'nproc='; nproc

echo '=== MEMORY ==='
free -h
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
cat /proc/pressure/memory 2>/dev/null || true

echo '=== FILESYSTEMS ==='
df -hT / /home /data
df -B1 / /home /data

echo '=== GPU_SUMMARY ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits

echo '=== GPU_PROCESSES ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits

echo '=== GPU_PROCESS_OWNERS ==='
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -nu); do
  ps -p "$pid" -o user=,pid=,ppid=,lstart=,etime=,%cpu=,%mem=,rss=,stat=,cmd= || true
done

echo '=== WRAPPERS ==='
ps -p 1477572,3589666 -o user=,pid=,ppid=,lstart=,etime=,%cpu=,%mem=,rss=,stat=,cmd= || true

echo '=== RAY_RLINF_COUNTS ==='
printf 'raylet='; pgrep -fc '[r]aylet' || true
printf 'gcs_server='; pgrep -fc '[g]cs_server' || true
printf 'RLinf='; pgrep -fc 'RLinf=' || true
printf 'RLinf_1='; pgrep -fc 'RLinf_1=' || true

echo '=== FASTWAM_RUN_FILES ==='
fwrun='/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1'
if [ -d "$fwrun" ]; then
  find "$fwrun" -maxdepth 3 -type f -printf '%T@\t%s\t%p\n' | sort -nr | head -n 25
  echo '--- FASTWAM STEP TAIL ---'
  find "$fwrun" -maxdepth 3 -type f \( -name '*.log' -o -name '*.txt' -o -name '*.out' \) -size -200M -print0 | xargs -0 -r grep -aEh 'Global Step|Step [0-9]+|success|fixed|KL|clip|grad|Traceback|CUDA out of memory|OOM|WorkerCrashed|nonfinite|NCCL|Vulkan|exit_code' 2>/dev/null | tail -n 120
else
  echo 'FASTWAM_RUN_MISSING'
fi

echo '=== PI05_RUN_FILES ==='
p05run='/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2'
if [ -d "$p05run" ]; then
  find "$p05run" -maxdepth 3 -type f -printf '%T@\t%s\t%p\n' | sort -nr | head -n 25
  echo '--- PI05 STEP TAIL ---'
  find "$p05run" -maxdepth 3 -type f \( -name '*.log' -o -name '*.txt' -o -name '*.out' \) -size -200M -print0 | xargs -0 -r grep -aEh 'Global Step|Step [0-9]+|success|fixed|KL|clip|grad|Traceback|CUDA out of memory|OOM|WorkerCrashed|nonfinite|NCCL|Vulkan|exit_code' 2>/dev/null | tail -n 120
else
  echo 'PI05_RUN_MISSING'
fi

echo '=== CHECKPOINTS ==='
find "$fwrun" "$p05run" -maxdepth 4 -type d -name 'global_step_*' -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -n 30

echo '=== FATAL_SCAN ==='
for run in "$fwrun" "$p05run"; do
  [ -d "$run" ] || continue
  printf '%s\t' "$run"
  find "$run" -maxdepth 3 -type f \( -name '*.log' -o -name '*.txt' -o -name '*.out' \) -size -200M -print0 | xargs -0 -r grep -aEic 'Traceback|CUDA out of memory|WorkerCrashedError|RayTaskError|NCCL.*(error|failed)|non.?finite|vk::|ErrorInitializationFailed|exit_code=[^0]' 2>/dev/null | awk '{s+=$1} END {print s+0}'
done

echo '=== USER_RESOURCE_AGGREGATES ==='
ps -eo user=,rss=,%cpu= | awk '{rss[$1]+=$2; cpu[$1]+=$3; n[$1]++} END {for (u in n) printf "%s\t%d\t%.1f\t%.3fGiB\n",u,n[u],cpu[u],rss[u]/1048576}' | sort -k4,4nr | head -n 20

echo '=== TOP_RSS_PROCESSES ==='
ps -eo user,pid,ppid,lstart,etime,%cpu,%mem,rss,stat,cmd --sort=-rss | head -n 35

echo '=== NETWORK ==='
ip route get 1.1.1.1 2>/dev/null | head -n 1 || true
ss -s 2>/dev/null || true
curl -L -sS -o /dev/null --max-time 10 -w 'github http=%{http_code} connect=%{time_connect}s total=%{time_total}s remote=%{remote_ip}\n' https://github.com/ || true
curl -L -sS -o /dev/null --max-time 10 -w 'hf http=%{http_code} connect=%{time_connect}s total=%{time_total}s remote=%{remote_ip}\n' https://huggingface.co/ || true

echo '=== LOGGED_IN_USERS ==='
who || true

