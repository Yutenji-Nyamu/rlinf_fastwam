#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
LOG="$RUN/runtime/driver.log"

section() { printf '\n===== %s =====\n' "$1"; }

section identity_time
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
hostname
id
uptime

section experiment_status
pid=$(cat "$RUN/runtime/wrapper.pid" 2>/dev/null || true)
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "wrapper_alive=yes pid=$pid"
else
  echo "wrapper_alive=no pid=${pid:-missing}"
fi
if [[ -f "$RUN/runtime/exit_code.txt" ]]; then printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"; else echo 'exit_code=pending'; fi
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$LOG" 2>/dev/null || true
printf '%s\n' '-- progress tail --'
grep -aE 'Global Step:|Generating Rollout Epochs:|success_once=' "$LOG" 2>/dev/null | tail -n 24 || true
printf '%s\n' '-- checkpoints --'
find "$RUN" -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -V || true
printf '%s\n' '-- eval/result/video files --'
find "$RUN" -type f \( -iname '*.mp4' -o -iname '*eval*' -o -iname '*success*' \) -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -k2,2 | tail -n 80 || true
printf '%s\n' '-- small artifact inventory --'
find "$RUN" -maxdepth 7 -type f \( -name 'events.out.tfevents*' -o -name 'resource.csv' -o -name 'resolved.yaml' -o -name 'driver.log' -o -name '*.json' \) -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -k2,2 || true
du -sh "$RUN" 2>/dev/null || true

section gpu_and_owners
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' '-- compute processes --'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
printf '%s\n' '-- pid owners --'
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  [[ -r "/proc/$p/status" ]] || continue
  ps -o user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,args= -p "$p"
done

section host_memory_and_processes
free -h
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree|AnonPages|Shmem|Slab|SReclaimable):' /proc/meminfo
printf '%s\n' '-- per-user aggregate --'
ps -eo user=,rss=,%cpu= | awk '{n[$1]++; rss[$1]+=$2; cpu[$1]+=$3} END {for (u in n) printf "%s processes=%d rss_gib=%.3f cpu_sum=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort
printf '%s\n' '-- top RSS processes --'
ps -eo user,pid,ppid,etimes,%cpu,%mem,rss,stat,comm,args --sort=-rss | head -n 32

section storage
df -hT / /home /data
df -ih / /home /data
findmnt -T / -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS
findmnt -T /home -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS
findmnt -T /data -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS

section sessions_and_recent_user_metadata
who -a || true
w -h || true
printf '%s\n' '-- active non-system users --'
ps -eo user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= --sort=-rss \
  | awk '$1 ~ /^(chenyiteng|liwenbo|xiongzizhen|zhangwei|toom|zhuanghuiping|guorenjie|qiufuwen)$/ {print}' \
  | head -n 80
printf '%s\n' '-- top-level recent metadata, last 24h --'
find /home /data -xdev -mindepth 1 -maxdepth 2 -mmin -1440 \
  -printf '%TY-%Tm-%TdT%TH:%TM %u %y %s %p\n' 2>/dev/null | sort -r | head -n 100 || true

section public_health
printf 'failed_units='; systemctl --failed --no-legend --no-pager 2>/dev/null | wc -l
printf 'mihomo='; systemctl is-active mihomo 2>/dev/null || true
printf 'docker='; systemctl is-active docker 2>/dev/null || true
printf 'memory_pressure='; tr '\n' ' ' </proc/pressure/memory; echo
printf 'io_pressure='; tr '\n' ' ' </proc/pressure/io; echo
printf 'cpu_pressure='; tr '\n' ' ' </proc/pressure/cpu; echo

echo SZ_DVAC_GRPO_V4_SERVER_READONLY_REFRESH_OK
