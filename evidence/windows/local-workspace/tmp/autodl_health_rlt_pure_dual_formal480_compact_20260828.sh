#!/usr/bin/env bash
set -u

pair=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_single_gpu_formal480_20260828_v1
s05=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1/runtime
s20=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1/runtime
date -Is
for spec in "s05:$s05" "s20:$s20"; do
  label=${spec%%:*}; runtime=${spec#*:}
  printf '%s ' "$label"
  test -f "$runtime/compose_exit_code.txt" && printf 'compose=%s ' "$(cat "$runtime/compose_exit_code.txt")"
  test -f "$runtime/started_at.txt" && printf 'started=%s ' "$(cat "$runtime/started_at.txt")"
  test -f "$runtime/exit_code.txt" && printf 'exit=%s ' "$(cat "$runtime/exit_code.txt")"
  if [ "$label" = s05 ]; then pattern=s0p5; else pattern=s2p0; fi
  printf 'driver_alive='
  pgrep -f "train_embodied_agent.py.*$pattern" >/dev/null 2>&1 && printf 'yes\n' || printf 'no\n'
  grep -a -E 'Global Step|Step [0-9]+|Generating Rollout Epochs|Gradient checkpointing|cache with size|ERROR|Traceback|CUDA out of memory|Killed' "$runtime/foreground.log" 2>/dev/null | tail -n 14 || true
done
printf '%s\n' 'GPU'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'memory_gib='
awk '{printf "%.2f\n",$1/1073741824}' /sys/fs/cgroup/memory.current
printf '%s\n' 'EVENTS'
cat /sys/fs/cgroup/memory.events
printf '%s\n' 'RAY_ACTORS'
ps -eo cmd | grep -E 'ray::(RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker)' | grep -v grep | sed 's/ .*//' | sort | uniq -c
printf '%s\n' 'RESOURCE_LAST'
tail -n 1 "$pair/paired_resources.csv"
