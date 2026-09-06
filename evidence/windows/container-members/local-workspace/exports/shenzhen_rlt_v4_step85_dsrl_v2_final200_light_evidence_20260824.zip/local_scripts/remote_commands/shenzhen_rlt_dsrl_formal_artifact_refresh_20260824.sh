#!/usr/bin/env bash
set -u

rlt_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
rlt_run="$rlt_root/robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_optimizer_warmup_v3"
rlt_runtime="$rlt_root/runtime"
dsrl_root=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2
dsrl_run="$dsrl_root/run"
stage1=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1/robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1

alive_state() {
  local file=$1 pid
  pid=$(cat "$file" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    printf 'pid=%s state=alive\n' "$pid"
  else
    printf 'pid=%s state=dead\n' "${pid:-missing}"
  fi
}

file_stat() {
  local file=$1
  if [ -e "$file" ]; then
    stat -c '%n|bytes=%s|mtime=%y' "$file"
  else
    printf '%s|missing\n' "$file"
  fi
}

printf '%s\n' '=== TIME ==='
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds

printf '%s\n' '=== OWNERS ==='
printf 'RLT '; alive_state "$rlt_runtime/wrapper.pid"
printf 'DSRL '; alive_state "$dsrl_run/wrapper.pid"

printf '%s\n' '=== LIVE PROCESS SUMMARY ==='
ps -eo pid,ppid,pgid,etimes,rss,stat,cmd --sort=pid \
  | grep -E 'formal-current-ar-stage2-8env250-20260824-v4-warmup-fix|formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2' \
  | grep -v grep | head -n 80 || true

printf '%s\n' '=== PROGRESS ==='
printf 'RLT completed_tables='; tr '\r' '\n' < "$rlt_runtime/driver.log" 2>/dev/null | grep -c 'Global Step:' || true
tr '\r' '\n' < "$rlt_runtime/driver.log" 2>/dev/null \
  | sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' \
  | grep -E 'Global Step:|success_once=|eval/success_once=|rlt/(replay_buffer_size|pending_update_budget|ready_for_online|update_step|min_replay_buffer_size)|Saving checkpoint' \
  | tail -n 100 || true
printf 'DSRL completed_tables='; tr '\r' '\n' < "$dsrl_run/driver.log" 2>/dev/null | grep -c 'Global Step:' || true
tr '\r' '\n' < "$dsrl_run/driver.log" 2>/dev/null \
  | sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' \
  | grep -E 'Global Step:|success_once=|eval/success_once=|dsrl/(global_resident_transitions|planned_optimizer_updates|update_step)|Saving checkpoint' \
  | tail -n 100 || true

printf '%s\n' '=== EXIT AND ERROR COUNTS ==='
for pair in "RLT:$rlt_runtime/driver.log" "DSRL:$dsrl_run/driver.log"; do
  name=${pair%%:*}; log=${pair#*:}
  printf '%s' "$name"
  for pattern in Traceback 'CUDA out of memory' OutOfMemory WorkerCrashed RayTaskError nonfinite NaN NCCL; do
    n=$(grep -a -i -c "$pattern" "$log" 2>/dev/null || true)
    printf '|%s=%s' "${pattern// /_}" "$n"
  done
  printf '\n'
done
file_stat "$rlt_runtime/chain_exit.txt"
file_stat "$rlt_runtime/exit_code.txt"
file_stat "$dsrl_run/exit_code.txt"

printf '%s\n' '=== LIGHTWEIGHT FILES ==='
for file in \
  "$stage1/driver.log" \
  "$rlt_runtime/driver.log" "$rlt_runtime/resource.csv" "$rlt_runtime/resolved.yaml" "$rlt_runtime/command.txt" \
  "$dsrl_run/driver.log" "$dsrl_run/resource.csv" "$dsrl_run/resolved.yaml" "$dsrl_run/command.txt"; do
  file_stat "$file"
done

printf '%s\n' '=== EVENT FILES ==='
find "$stage1" "$rlt_root" "$dsrl_root" -type f -name 'events.out.tfevents*' \
  -printf '%s|%TY-%Tm-%TdT%TH:%TM:%TS|%p\n' 2>/dev/null | sort -t'|' -k3,3 || true

printf '%s\n' '=== CHECKPOINT DIRECTORIES ==='
for root in "$stage1/checkpoints" "$rlt_run/checkpoints" "$dsrl_root/checkpoints"; do
  printf 'root=%s\n' "$root"
  find "$root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print0 2>/dev/null \
    | sort -zV \
    | while IFS= read -r -d '' dir; do
        bytes=$(du -sb "$dir" 2>/dev/null | awk '{print $1}')
        files=$(find "$dir" -type f 2>/dev/null | wc -l)
        printf '%s|bytes=%s|files=%s\n' "$dir" "${bytes:-0}" "$files"
      done
done

printf '%s\n' '=== CHECKPOINT MANIFESTS AND SIDECARS ==='
find "$rlt_run/checkpoints" "$dsrl_root/checkpoints" -type f \
  \( -name 'artifact_manifest.json' -o -name '*algorithm_state*.json' -o -name '*runtime_state*.json' -o -name '*manifest*.json' \) \
  -printf '%s|%p\n' 2>/dev/null | sort -t'|' -k2,2 || true

printf '%s\n' '=== OUTPUT CLASS COUNTS ==='
for root in "$stage1" "$rlt_root" "$dsrl_root"; do
  printf 'root=%s|bytes=' "$root"; du -sb "$root" 2>/dev/null | awk '{print $1}'
  for ext in mp4 json jsonl csv yaml log pt safetensors distcp; do
    printf '%s=%s ' "$ext" "$(find "$root" -type f -name "*.$ext" 2>/dev/null | wc -l)"
  done
  printf '\n'
done
for dir in "$rlt_root/robotwin_data" "$rlt_root/video" "$dsrl_root/robotwin_data" "$dsrl_root/video"; do
  if [ -d "$dir" ]; then
    printf '%s|bytes=' "$dir"; du -sb "$dir" 2>/dev/null | awk '{print $1}'
    printf '%s|files=' "$dir"; find "$dir" -type f 2>/dev/null | wc -l
  else
    printf '%s|missing\n' "$dir"
  fi
done

printf '%s\n' '=== RESOURCE SCHEMA AND TAIL ==='
printf 'RLT rows='; wc -l < "$rlt_runtime/resource.csv" 2>/dev/null || true
head -n 1 "$rlt_runtime/resource.csv" 2>/dev/null || true
tail -n 3 "$rlt_runtime/resource.csv" 2>/dev/null || true
printf 'DSRL rows='; wc -l < "$dsrl_run/resource.csv" 2>/dev/null || true
head -n 1 "$dsrl_run/resource.csv" 2>/dev/null || true
tail -n 3 "$dsrl_run/resource.csv" 2>/dev/null || true

printf '%s\n' '=== CURRENT RESOURCES ==='
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits
free -b | sed -n '1,2p'
df -B1 / /home /data | tail -n +2
