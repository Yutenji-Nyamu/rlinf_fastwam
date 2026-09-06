#!/usr/bin/env bash
set -euo pipefail

PARENT=/data/chenyiteng/results/dvac-observation/run-metadata/fastwam-multitask-p2-3x16-c63dc9b5-v1
TASK=move_stapler_pad
RUN_ID=fastwam-${TASK}-p2-16ep-c63dc9b5-v1
META=/data/chenyiteng/results/dvac-observation/run-metadata/$RUN_ID
PAYLOAD=/data/chenyiteng/results/dvac-observation/$RUN_ID
PID=$(cat "$PARENT/driver.pid")

printf 'time=%s\n' "$(date --iso-8601=seconds)"
if kill -0 "$PID" 2>/dev/null; then printf 'parent_alive=1\n'; else printf 'parent_alive=0\n'; fi
printf 'parent_pid=%s\n' "$PID"
printf 'child_pids='; pgrep -P "$PID" | tr '\n' ',' || true; printf '\n'
nvidia-smi -i 3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'gpu3_compute='; nvidia-smi -i 3 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr '\n' ';'; printf '\n'
awk '/MemAvailable:/ {print}' /proc/meminfo
printf 'fatal_count=%s\n' "$(grep -Eic 'Traceback|CUDA out of memory|illegal instruction|SIGSEGV|RuntimeError|Evaluation failed' "$PARENT/driver.log" "$META/driver.log" 2>/dev/null || true)"
if test -f "$PAYLOAD/episodes.csv"; then printf 'episode_rows=%s\n' "$(( $(wc -l < "$PAYLOAD/episodes.csv") - 1 ))"; else printf 'episode_rows=0\n'; fi
if test -f "$PAYLOAD/queries.csv"; then printf 'query_rows=%s\n' "$(( $(wc -l < "$PAYLOAD/queries.csv") - 1 ))"; else printf 'query_rows=0\n'; fi
printf '%s\n' '--- parent tail ---'
tail -n 12 "$PARENT/driver.log"
printf '%s\n' '--- task tail ---'
tail -n 30 "$META/driver.log" 2>/dev/null || true
