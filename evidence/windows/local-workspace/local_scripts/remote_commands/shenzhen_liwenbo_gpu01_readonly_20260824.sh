#!/usr/bin/env bash
set -u

echo '=== identity/time ==='
date '+%F %T %Z'
id

echo '=== GPU 0/1 summary ==='
nvidia-smi -i 0,1 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
nvidia-smi -i 0,1 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits

echo '=== GPU 0/1 process identity ==='
mapfile -t pids < <(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
for pid in "${pids[@]}"; do
  echo "--- pid=$pid ---"
  ps -o user=,uid=,pid=,ppid=,pgid=,lstart=,etime=,rss=,stat=,args= -p "$pid" 2>/dev/null || true
  if [[ -r "/proc/$pid/cmdline" ]]; then tr '\0' ' ' < "/proc/$pid/cmdline"; echo; fi
  readlink -f "/proc/$pid/cwd" 2>/dev/null | sed 's/^/cwd=/' || true
done

echo '=== liwenbo process tree, bounded ==='
ps -u liwenbo -o pid=,ppid=,pgid=,lstart=,etime=,%cpu=,%mem=,rss=,stat=,args= --sort=-rss | sed -n '1,120p'

