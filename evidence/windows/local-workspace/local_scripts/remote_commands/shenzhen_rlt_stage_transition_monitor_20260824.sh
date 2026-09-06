#!/usr/bin/env bash
set -u

root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2
runtime=$root/runtime
stage1=$root/stage1
stage2=$root/stage2

date --iso-8601=seconds
printf '%s\n' '--- processes ---'
for item in wrapper resource_observer; do
  file="$runtime/$item.pid"
  if [ -s "$file" ]; then
    pid=$(cat "$file")
    if kill -0 "$pid" 2>/dev/null; then state=alive; else state=dead; fi
    printf '%s pid=%s state=%s\n' "$item" "$pid" "$state"
  fi
done
printf '%s\n' '--- stage1 completion ---'
for file in "$stage1/runtime/exit_code.txt" "$stage1/runtime/finished_at.txt"; do
  if [ -s "$file" ]; then printf '%s=' "$file"; tr '\n' ' ' < "$file"; printf '\n'; fi
done
find "$stage1" -type d -name 'global_step_2000' -print 2>/dev/null | head -n 5
printf '%s\n' '--- stage2 markers ---'
for file in "$stage2/runtime/started_at.txt" "$stage2/runtime/driver.pid" "$stage2/runtime/exit_code.txt"; do
  if [ -s "$file" ]; then printf '%s=' "$file"; tr '\n' ' ' < "$file"; printf '\n'; fi
done
printf '%s\n' '--- stage2 selected lines ---'
grep -aE 'Global Step|Global step|cycle|Cycle|rollout|trajectory|transition|update_step|success|Traceback|ERROR|Error|Exception|OOM' "$stage2/runtime/driver.log" 2>/dev/null | tail -n 40 || true
printf '%s\n' '--- stage2 tail ---'
tail -n 40 "$stage2/runtime/driver.log" 2>/dev/null || true
printf '%s\n' '--- gpu4-5 process table ---'
nvidia-smi -i 4,5 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
