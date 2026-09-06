#!/usr/bin/env bash
set -euo pipefail

echo '=== TIME_UPTIME ==='
date --iso-8601=seconds
uptime

echo '=== MEMORY_PSI ==='
free -h
cat /proc/pressure/memory

echo '=== FILESYSTEM ==='
df -h / /home /data
df -ih / /home /data

echo '=== GPU ==='
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
echo '=== GPU_PROCESSES ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader || true

echo '=== RAY ==='
ps -eo user:16,pid,ppid,pgid,etimes,rss,pcpu,cmd --sort=-rss | grep -E 'raylet|gcs_server|train_embodied_agent|fastwam-grpo-control|pi05-grpo-control|run_.*formal|monitor_.*formal' | grep -v grep | head -80 || true

echo '=== CURRENT_RUN_DIRS ==='
for root in \
  /data/chenyiteng/results/rlinf-shenzhen/pi05-grpo/runs \
  /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs; do
  echo "ROOT=$root"
  if [ -d "$root" ]; then
    find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' | sort -nr | head -8
  fi
done

echo '=== LIVE_RUNS ==='
for run in \
  /data/chenyiteng/results/rlinf-shenzhen/pi05-grpo/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2 \
  /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1; do
  echo "RUN=$run"
  if [ ! -d "$run" ]; then echo 'MISSING'; continue; fi
  find "$run" -maxdepth 3 -type f \( -name 'driver.log' -o -name 'resource.csv' -o -name 'resolved.yaml' -o -name 'exit_code' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort
  driver=$(find "$run" -maxdepth 3 -type f -name driver.log | head -1)
  if [ -n "$driver" ]; then
    echo "DRIVER=$driver"
    grep -aE 'Global Step|Train.*success|eval.*success|Success|KL|clip|grad|Fixed|Generating Rollout|Saving checkpoint|checkpoint' "$driver" | tail -80 || true
    echo 'FATAL_COUNT'
    grep -aEic 'Traceback|CUDA out of memory|OutOfMemory|nonfinite|nan|RayActorError|WorkerCrashedError|ErrorInitializationFailed|NCCL.*error|fatal' "$driver" || true
    echo 'TAIL'
    tail -40 "$driver"
  fi
  echo 'CHECKPOINTS'
  find "$run" -maxdepth 4 -type d -name 'global_step_*' -printf '%T@ %p\n' | sort -n | tail -8 || true
  echo 'EVENTS'
  find "$run" -maxdepth 4 -type f -name 'events.out.tfevents.*' -printf '%T@ %s %p\n' | sort -n | tail -4 || true
done

echo '=== OTHER_USER_ACTIVITY ==='
ps -eo user:16,pid,etimes,rss,pcpu,pmem,cmd --sort=-rss | awk 'NR==1 || ($1!="chenyiteng" && $1!="root" && $1!="ray")' | head -35

echo '=== NETWORK ==='
for url in https://github.com https://huggingface.co; do
  code=$(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy curl -LIsS --max-time 8 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true)
  echo "$url direct=$code"
done
systemctl is-active mihomo 2>/dev/null || true
