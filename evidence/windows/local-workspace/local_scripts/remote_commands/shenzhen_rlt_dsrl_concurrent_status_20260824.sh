#!/usr/bin/env bash
set -u

rlt=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2
dsrl=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run

date --iso-8601=seconds
printf '%s\n' '--- owner state ---'
for pair in \
  "rlt:$rlt/runtime/wrapper.pid" \
  "dsrl:$dsrl/wrapper.pid"; do
  name=${pair%%:*}; file=${pair#*:}; pid=$(cat "$file" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state=alive; else state=dead; fi
  printf '%s pid=%s state=%s\n' "$name" "${pid:-missing}" "$state"
done

printf '%s\n' '--- RLT latest progress ---'
tr '\r' '\n' < "$rlt/stage1/runtime/driver.log" 2>/dev/null \
  | grep 'Global Step:' | tail -n 3 || true

printf '%s\n' '--- DSRL namespace/start/progress ---'
grep -E 'namespace conflict|RLinf is running|Saving checkpoint|Global Step:|resident_transitions|planned_optimizer_updates|success_once|episode_len=|return=' \
  "$dsrl/driver.log" 2>/dev/null | tail -n 70 || true
printf '%s\n' '--- DSRL tail ---'
tail -n 40 "$dsrl/driver.log" 2>/dev/null || true

printf '%s\n' '--- GPUs 4-7 ---'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' '--- GPU process map ---'
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
printf '%s\n' '--- host memory ---'
free -h | sed -n '1,2p'
printf '%s\n' '--- observer tails ---'
tail -n 4 "$rlt/runtime/resource.csv" 2>/dev/null || true
tail -n 4 "$dsrl/resource.csv" 2>/dev/null || true
printf '%s\n' '--- exit markers ---'
for file in "$rlt/runtime/chain_exit.txt" "$dsrl/exit_code.txt"; do
  if [ -s "$file" ]; then printf '%s=' "$file"; tr '\n' ' ' < "$file"; printf '\n'; fi
done
