#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1
LOG="$RUN/runtime/driver.log"

section() { printf '\n===== %s =====\n' "$1"; }

section identity_time
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
hostname
id
uptime

section w0to5_runtime
pid=$(cat "$RUN/runtime/wrapper.pid" 2>/dev/null || true)
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "wrapper_alive=yes pid=$pid"
  ps -o user=,pid=,ppid=,pgid=,etimes=,%cpu=,%mem=,rss=,stat=,args= -p "$pid" || true
else
  echo "wrapper_alive=no pid=${pid:-missing}"
fi
if [[ -f "$RUN/runtime/exit_code.txt" ]]; then
  printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"
else
  echo 'exit_code=pending'
fi
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$LOG" 2>/dev/null || true
printf '%s\n' '-- latest progress --'
grep -aE 'Global Step:|Generating Rollout Epochs:|success_once=' "$LOG" 2>/dev/null | tail -n 36 || true
printf '%s\n' '-- latest optimizer and DVAC metrics --'
grep -aEi 'approx.*kl|clip.*fraction|grad.*norm|dvac|weight_(mean|std|min|max)|effective.*sample|ess' "$LOG" 2>/dev/null | tail -n 80 || true
printf '%s\n' '-- checkpoints --'
find "$RUN" -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -V || true
printf '%s\n' '-- runtime small files --'
find "$RUN/runtime" -maxdepth 1 -type f -printf '%f %s %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort || true
printf '%s\n' '-- latest resource rows --'
tail -n 6 "$RUN/runtime/resource.csv" 2>/dev/null || true

section gpu_and_owners
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' '-- compute processes --'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
printf '%s\n' '-- compute pid owners --'
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  [[ -r "/proc/$p/status" ]] || continue
  ps -o user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= -p "$p" || true
done

section host_health
free -h
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree|AnonPages|Shmem|Slab|SReclaimable):' /proc/meminfo
printf '%s\n' '-- pressure --'
printf 'memory '; tr '\n' ' ' </proc/pressure/memory; echo
printf 'io '; tr '\n' ' ' </proc/pressure/io; echo
printf 'cpu '; tr '\n' ' ' </proc/pressure/cpu; echo
printf '%s\n' '-- failed units --'
systemctl --failed --no-pager --no-legend 2>/dev/null || true

section storage
df -hT / /home /data
df -ih / /home /data
findmnt -T / -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS
findmnt -T /home -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS
findmnt -T /data -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS

section network
printf 'mihomo_active='; systemctl is-active mihomo 2>/dev/null || true
systemctl show mihomo -p ActiveEnterTimestamp -p NRestarts --no-pager 2>/dev/null || true
ss -lnt 2>/dev/null | awk '$4 ~ /127\.0\.0\.1:(7890|9090)$/ {print}' || true
for spec in \
  'direct_github https://github.com/' \
  'direct_hf https://huggingface.co/api/models?limit=1'; do
  set -- $spec
  name=$1; url=$2
  code=$(curl --noproxy '*' -L -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 12 "$url" 2>/dev/null || true)
  echo "$name=$code"
done
for spec in \
  'proxy_github https://github.com/' \
  'proxy_hf https://huggingface.co/api/models?limit=1'; do
  set -- $spec
  name=$1; url=$2
  code=$(curl --proxy http://127.0.0.1:7890 -L -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 12 "$url" 2>/dev/null || true)
  echo "$name=$code"
done

section other_users_public_processes
printf '%s\n' '-- sessions --'
who -a 2>/dev/null || true
w -h 2>/dev/null || true
printf '%s\n' '-- per-user aggregates --'
ps -eo user=,rss=,%cpu= | awk '$1 ~ /^(chenyiteng|liwenbo|xiongzizhen|zhangwei|toom|zhuanghuiping|guorenjie|qiufuwen|yanchuhan)$/ {n[$1]++; rss[$1]+=$2; cpu[$1]+=$3} END {for (u in n) printf "%s processes=%d rss_gib=%.3f cpu_sum=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort
printf '%s\n' '-- public process summaries --'
ps -eo user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= --sort=-rss \
  | awk '$1 ~ /^(chenyiteng|liwenbo|xiongzizhen|zhangwei|toom|zhuanghuiping|guorenjie|qiufuwen|yanchuhan)$/ {print}' \
  | head -n 100

echo SZ_W0TO5_SERVER_READONLY_REFRESH_OK
