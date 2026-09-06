#!/usr/bin/env bash
set -u

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
ROOT=/data/chenyiteng/results/dvac-observation
PARENT_ID=fastwam-multitask-p2-3x16-c63dc9b5-v1
PARENT_META="$ROOT/run-metadata/$PARENT_ID"
PARENT_PID=819933
TASKS=(move_stapler_pad turn_switch pick_diverse_bottles)

printf 'TIME=%s\n' "$(date --iso-8601=seconds)"
if kill -0 "$PARENT_PID" 2>/dev/null; then
  printf 'PARENT_ALIVE=1\n'
else
  printf 'PARENT_ALIVE=0\n'
fi
ps -o pid=,ppid=,stat=,etime=,rss=,cmd= --forest -g "$PARENT_PID" 2>/dev/null || true

printf '%s\n' 'GPU3_BEGIN'
nvidia-smi -i 3 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 3 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true
printf '%s\n' 'GPU3_END'
awk '/MemAvailable:/ {print "MEM_AVAILABLE_KIB=" $2}' /proc/meminfo
df -B1 --output=avail /data | tail -n 1 | awk '{print "DATA_AVAILABLE_BYTES=" $1}'

for task in "${TASKS[@]}"; do
  run_id="fastwam-${task}-p2-16ep-c63dc9b5-v1"
  payload="$ROOT/$run_id"
  meta="$ROOT/run-metadata/$run_id"
  official="$WT/evaluate_results/robotwin/robotwin_uncond_3cam_384/$run_id"
  episodes=0
  success=0
  queries=0
  npz=0
  png=0
  mp4=0
  fatal=0
  if test -s "$payload/episodes.csv"; then
    episodes=$(( $(wc -l < "$payload/episodes.csv") - 1 ))
    success=$(tail -n +2 "$payload/episodes.csv" | awk -F, '{for(i=1;i<=NF;i++) if(tolower($i)=="true") {n++; break}} END {print n+0}')
  fi
  if test -s "$payload/queries.csv"; then
    queries=$(( $(wc -l < "$payload/queries.csv") - 1 ))
  fi
  if test -d "$payload"; then
    npz=$(find "$payload" -type f -name '*.npz' | wc -l)
    png=$(find "$payload" -type f -name '*.png' | wc -l)
  fi
  if test -d "$official"; then
    mp4=$(find "$official" -type f -name '*.mp4' -size +1024c | wc -l)
  fi
  if test -s "$meta/driver.log"; then
    fatal=$(grep -Eic 'Traceback|CUDA out of memory|illegal instruction|RuntimeError|Error executing job|Evaluation failed' "$meta/driver.log" || true)
  fi
  printf 'TASK=%s EPISODES=%s SUCCESS=%s QUERIES=%s NPZ=%s PNG=%s MP4=%s FATAL=%s\n' \
    "$task" "$episodes" "$success" "$queries" "$npz" "$png" "$mp4" "$fatal"
done

printf '%s\n' 'PARENT_TAIL_BEGIN'
tail -n 20 "$PARENT_META/driver.log" 2>/dev/null || true
printf '%s\n' 'PARENT_TAIL_END'

for task in "${TASKS[@]}"; do
  run_id="fastwam-${task}-p2-16ep-c63dc9b5-v1"
  meta="$ROOT/run-metadata/$run_id"
  if test -s "$meta/driver.log"; then
    printf 'ACTIVE_TAIL_TASK=%s\n' "$task"
    tail -n 12 "$meta/driver.log"
  fi
done
