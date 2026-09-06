#!/usr/bin/env bash
set -euo pipefail

PI0=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1
PID=$(cat "$PI0/driver.pid")
printf 'time=%s\n' "$(date --iso-8601=seconds)"
if kill -0 "$PID" 2>/dev/null; then printf 'pi0_owned_pid_alive=1\n'; else printf 'pi0_owned_pid_alive=0\n'; fi
printf 'raylet_count=%s\n' "$(pgrep -u "$(id -u)" -x raylet | wc -l)"
printf 'gcs_count=%s\n' "$(pgrep -u "$(id -u)" -x gcs_server | wc -l)"
printf 'gpu3_compute_pids='; nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits | tr '\n' ','; printf '\n'
nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
awk '/MemAvailable:/ {print}' /proc/meminfo
df -B1 --output=avail /data | tail -n 1
printf 'fatal_count=%s\n' "$(grep -Eic 'Traceback|CUDA out of memory|illegal instruction|SIGSEGV|worker.*died|RayActorError' "$PI0/driver.log" || true)"
tail -n 8 "$PI0/driver.log"
