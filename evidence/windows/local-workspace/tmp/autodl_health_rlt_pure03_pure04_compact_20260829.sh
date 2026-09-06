#!/usr/bin/env bash
set -u

pair=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_dual_single_gpu_formal480_20260829_v1
p03=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1/runtime
p04=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1/runtime
date -Is
for spec in "pure03:$p03:s1p0" "pure04:$p04:s1p5"; do
  label=${spec%%:*}; rest=${spec#*:}; runtime=${rest%%:*}; pattern=${rest##*:}
  printf '%s ' "$label"
  test -f "$runtime/compose_exit_code.txt" && printf 'compose=%s ' "$(cat "$runtime/compose_exit_code.txt")"
  test -f "$runtime/started_at.txt" && printf 'started=%s ' "$(cat "$runtime/started_at.txt")"
  test -f "$runtime/exit_code.txt" && printf 'exit=%s ' "$(cat "$runtime/exit_code.txt")"
  printf 'driver_alive='; pgrep -f "train_embodied_agent.py.*$pattern" >/dev/null 2>&1 && printf 'yes\n' || printf 'no\n'
  grep -a -E 'Global Step|Generating Rollout Epochs|ERROR|Traceback|CUDA out of memory|Killed' "$runtime/foreground.log" 2>/dev/null | tail -n 10 || true
done
printf '%s\n' 'GPU'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'memory_gib='; awk '{printf "%.2f\n",$1/1073741824}' /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
printf '%s\n' 'RAY_ACTORS'
ps -eo cmd | grep -E 'ray::(RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker)' | grep -v grep | sed 's/ .*//' | sort | uniq -c
test -f "$pair/paired_resources.csv" && { printf '%s\n' 'RESOURCE_LAST'; tail -n 1 "$pair/paired_resources.csv"; }
