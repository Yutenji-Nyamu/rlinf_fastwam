#!/usr/bin/env bash
set -u

pair=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_single_gpu_formal480_20260828_v1
s05=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1/runtime
s20=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1/runtime

date -Is
hostname
cat "$pair/launch_summary.txt"
printf '%s\n' 'PROCESS_SNAPSHOT'
ps -eo pid,ppid,etimes,cmd | grep -E 'train_embodied_agent.py|ray::|raylet|gcs_server' | grep -v grep | tail -n 80
printf '%s\n' 'GPU_SNAPSHOT'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory_current_bytes='
cat /sys/fs/cgroup/memory.current
printf '%s\n' 'MEMORY_EVENTS'
cat /sys/fs/cgroup/memory.events
for label in s05 s20; do
  if [ "$label" = s05 ]; then runtime="$s05"; else runtime="$s20"; fi
  printf '%s\n' "${label}_RUNTIME"
  test -f "$runtime/compose_exit_code.txt" && printf 'compose_exit=' && cat "$runtime/compose_exit_code.txt"
  test -f "$runtime/started_at.txt" && printf 'started_at=' && cat "$runtime/started_at.txt"
  test -f "$runtime/exit_code.txt" && printf 'exit_code=' && cat "$runtime/exit_code.txt"
  test -f "$runtime/resolved.sha256" && cat "$runtime/resolved.sha256"
  printf '%s\n' "${label}_LOG_TAIL"
  tail -n 35 "$runtime/foreground.log" 2>/dev/null || true
done
printf '%s\n' 'RECENT_ERRORS'
grep -R -n -E 'Traceback|CUDA out of memory|OutOfMemory|OOM|Killed|FATAL|Fatal' "$s05" "$s20" 2>/dev/null | tail -n 30 || true
printf '%s\n' 'RESOURCE_TAIL'
tail -n 5 "$pair/paired_resources.csv" 2>/dev/null || true
