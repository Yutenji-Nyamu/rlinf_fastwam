#!/usr/bin/env bash
set -u

rlt=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3/runtime
dsrl=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run

date --iso-8601=seconds
printf '%s\n' '--- owner state ---'
for pair in "rlt:$rlt/wrapper.pid" "dsrl:$dsrl/wrapper.pid"; do
  name=${pair%%:*}; file=${pair#*:}; pid=$(cat "$file" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state=alive; else state=dead; fi
  printf '%s pid=%s state=%s\n' "$name" "${pid:-missing}" "$state"
done

printf '%s\n' '--- RLT Stage2 v3 progress ---'
tr '\r' '\n' < "$rlt/driver.log" 2>/dev/null \
  | grep -E 'Global Step:|collected [0-9]+ trajectories|replay_buffer|warm-up|update_step|Saving checkpoint|success_once' \
  | tail -n 45 || true
printf '%s\n' '--- DSRL v2 progress ---'
tr '\r' '\n' < "$dsrl/driver.log" 2>/dev/null \
  | grep -E 'Global Step:|collected [0-9]+ trajectories|resident_transitions|planned_optimizer_updates|replay warm-up|Saving checkpoint|success_once' \
  | tail -n 55 || true

printf '%s\n' '--- isolated resolved outputs ---'
grep -nE 'save_path:|video_base_dir:' "$rlt/resolved.yaml" 2>/dev/null | head -n 8 || true
grep -nE 'save_path:|video_base_dir:' "$dsrl/resolved.yaml" 2>/dev/null | head -n 8 || true

printf '%s\n' '--- GPUs 4-7 ---'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' '--- GPU process map ---'
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
printf '%s\n' '--- host memory ---'
free -h | sed -n '1,2p'
printf '%s\n' '--- observer tails ---'
tail -n 3 "$rlt/resource.csv" 2>/dev/null || true
tail -n 3 "$dsrl/resource.csv" 2>/dev/null || true
printf '%s\n' '--- stale actor check ---'
stale=0
for pid in 344717 344719 344723 344725 344727 344782 364390 364392 364393 364395 364396 364398; do
  if kill -0 "$pid" 2>/dev/null; then printf 'stale_pid=%s state=alive\n' "$pid"; stale=$((stale + 1)); fi
done
printf 'stale_actor_count=%s\n' "$stale"
printf '%s\n' '--- exit markers ---'
for file in "$rlt/exit_code.txt" "$dsrl/exit_code.txt"; do
  if [ -s "$file" ]; then printf '%s=' "$file"; tr '\n' ' ' < "$file"; printf '\n'; fi
done
