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
  printf 'progress\n'
  grep -a -E 'Global Step:|Generating Rollout Epochs' "$runtime/foreground.log" 2>/dev/null | tail -n 8 || true
  printf 'fatal_counts '
  for fatal in 'CUDA out of memory' 'OutOfMemoryError' 'NCCL.*(error|Error)' 'RayTaskError' 'worker fatal' 'OOMKilled'; do
    count=$(grep -a -E -c "$fatal" "$runtime/foreground.log" 2>/dev/null || true)
    printf '%s=%s ' "$fatal" "$count"
  done
  printf '\n'
done

printf '%s\n' 'GPU index,memory_used_mib,util_pct'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'memory_gib='; awk '{printf "%.2f\n",$1/1073741824}' /sys/fs/cgroup/memory.current
printf 'memory_events '; tr '\n' ' ' </sys/fs/cgroup/memory.events; printf '\n'
printf '%s\n' 'RAY_ACTORS'
ps -eo cmd | grep -E 'ray::(RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker)' | grep -v grep | sed 's/ .*//' | sort | uniq -c
if test -f "$pair/paired_resources.csv"; then
  printf '%s\n' 'RESOURCE_LAST'
  tail -n 1 "$pair/paired_resources.csv"
  printf '%s\n' 'RESOURCE_PEAKS'
  awk -F, 'NR==1 {for(i=1;i<=NF;i++) h[i]=$i; next} {for(i=1;i<=NF;i++) if(($i+0)>m[i]) m[i]=$i+0} END {for(i=1;i<=NF;i++) printf "%s=%s%s",h[i],m[i],(i==NF?"\n":" ")}' "$pair/paired_resources.csv"
fi

for run in \
  /root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1 \
  /root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1; do
  printf 'checkpoints %s: ' "$run"
  find "$run" -maxdepth 2 -type d -name 'global_step_*' -printf '%f ' 2>/dev/null | sort -V | tail -n 6
  printf '\n'
done
