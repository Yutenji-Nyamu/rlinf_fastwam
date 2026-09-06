#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1
PID=$(cat "$RUN/driver.pid")

printf 'time=%s\n' "$(date --iso-8601=seconds)"
if kill -0 "$PID" 2>/dev/null; then printf 'owned_pid_alive=1\n'; else printf 'owned_pid_alive=0\n'; fi
printf 'raylet_count=%s\n' "$(pgrep -u "$(id -u)" -x raylet | wc -l)"
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'fatal_count=%s\n' "$(grep -Eic 'Traceback|CUDA out of memory|illegal instruction|SIGSEGV|worker.*died|RayActorError' "$RUN/driver.log" || true)"
printf 'global_step_lines=%s\n' "$(grep -c 'Global Step' "$RUN/driver.log" || true)"
printf 'episode_csv_rows='; find "$RUN" -type f -name 'episode_index_env_rank*.csv' -print0 | xargs -0 -r awk 'FNR>1 {n++} END {print n+0}'
printf 'query_csv_rows='; find "$RUN" -type f -name 'query_index_rollout_rank*.csv' -print0 | xargs -0 -r awk 'FNR>1 {n++} END {print n+0}'
printf 'npz_count=%s\n' "$(find "$RUN" -type f -name '*.npz' | wc -l)"
printf 'png_count=%s\n' "$(find "$RUN" -type f -name '*.png' | wc -l)"
printf 'mp4_count=%s\n' "$(find "$RUN" -type f -name '*.mp4' | wc -l)"
tail -n 40 "$RUN/driver.log"
