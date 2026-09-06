set -euo pipefail
control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v5/runtime
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v5/runtime
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v5

date -Is
echo '[PROCESS]'
for runtime in "$control" "$method"; do
  if test -f "$runtime/wrapper.pid"; then
    pid=$(cat "$runtime/wrapper.pid")
    ps -p "$pid" -o pid=,pgid=,stat=,etime=,cmd= || true
  fi
done
echo '[GPU]'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '[MEMORY]'
printf 'current='; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
echo '[CONTROL_MARKERS]'
grep -E 'Global Step|Evaluation|Traceback|ERROR|Error|CUDA out of memory|Killed|Actor Loss|Critic Loss' "$control/foreground.log" 2>/dev/null | tail -20 || true
echo '[METHOD_MARKERS]'
grep -E 'Global Step|Evaluation|Traceback|ERROR|Error|CUDA out of memory|Killed|Actor Loss|Critic Loss|rlt_dvac' "$method/foreground.log" 2>/dev/null | tail -25 || true
echo '[EXIT]'
for runtime in "$control" "$method"; do
  if test -f "$runtime/exit_code.txt"; then
    printf '%s=' "$runtime"; cat "$runtime/exit_code.txt"
  fi
done
echo '[RESOURCE_TAIL]'
tail -3 "$pair/paired_resources.csv" 2>/dev/null || true
