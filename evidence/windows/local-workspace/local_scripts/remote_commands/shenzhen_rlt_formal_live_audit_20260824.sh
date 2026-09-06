#!/usr/bin/env bash
set -u

stage1=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1/robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1
stage2=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3
runtime=$stage2/runtime

date --iso-8601=seconds
printf '%s\n' '--- wrapper and actors ---'
pid=$(cat "$runtime/wrapper.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state=alive; else state=dead; fi
printf 'wrapper_pid=%s wrapper_state=%s\n' "${pid:-missing}" "$state"
ps -eo pid,ppid,etimes,rss,cmd --sort=pid \
  | grep -E 'formal-current-ar-stage2-8env250-20260824-v3|RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker' \
  | grep -v grep || true

printf '%s\n' '--- stage1 terminal evidence ---'
find "$stage1" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null \
  | sort | tail -n 30 || true
find "$stage1/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -V || true
du -sh "$stage1/checkpoints/global_step_2000" 2>/dev/null || true

printf '%s\n' '--- stage2 runtime files ---'
for file in "$runtime/driver.log" "$runtime/resource.csv" "$runtime/resolved.yaml" "$runtime/launcher.sh" "$runtime/wrapper.pid" "$runtime/exit_code.txt"; do
  if [ -e "$file" ]; then stat -c '%y %s %n' "$file"; else printf 'missing %s\n' "$file"; fi
done

printf '%s\n' '--- stage2 resource schema and extent ---'
head -n 2 "$runtime/resource.csv" 2>/dev/null || true
wc -l "$runtime/resource.csv" 2>/dev/null || true
tail -n 3 "$runtime/resource.csv" 2>/dev/null || true

printf '%s\n' '--- stage2 full completed-step sequence ---'
tr '\r' '\n' < "$runtime/driver.log" 2>/dev/null \
  | sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' \
  | grep -E 'Global Step:' || true

printf '%s\n' '--- stage2 latest metric blocks ---'
tr '\r' '\n' < "$runtime/driver.log" 2>/dev/null \
  | sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' \
  | grep -E 'Global Step:|reward=|success_once=|rlt/(replay_buffer_size|pending_update_budget|ready_for_online|update_step|min_replay_buffer_size)|Saving checkpoint|Eval' \
  | tail -n 140 || true

printf '%s\n' '--- stage2 targeted errors ---'
for pattern in 'Traceback' 'CUDA out of memory' 'OutOfMemoryError' 'WorkerCrashedError' 'nonfinite' 'NaN' 'Killed'; do
  count=$(grep -a -F -c "$pattern" "$runtime/driver.log" 2>/dev/null || true)
  printf '%s=%s\n' "$pattern" "$count"
done

printf '%s\n' '--- stage2 checkpoints, events, videos, data ---'
find "$stage2" -maxdepth 5 -type f \
  \( -name 'events.out.tfevents*' -o -name '*.pt' -o -name '*.safetensors' -o -name '*.json' -o -name '*.yaml' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 100 || true
find "$stage2" -type f -name '*.mp4' -printf '.' 2>/dev/null | wc -c | awk '{print "mp4_count=" $1}'
find "$stage2" -type f -name '*.mp4' -printf '%s\n' 2>/dev/null | awk '{n+=1;s+=$1} END{print "mp4_bytes=" s+0}'
find "$stage2/robotwin_data" -type f -printf '.' 2>/dev/null | wc -c | awk '{print "robotwin_data_file_count=" $1}'
find "$stage2/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -V || true

printf '%s\n' '--- current GPU and host ---'
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits
free -b | sed -n '1,2p'
df -B1 /data | tail -n 1

