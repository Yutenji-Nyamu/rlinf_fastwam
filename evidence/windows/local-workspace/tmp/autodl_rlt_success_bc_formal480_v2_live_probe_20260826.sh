set -euo pipefail

control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_formal480_20260825_v2
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2
control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v2/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v2

echo TIME
date -Is
echo PROCESSES
for root in "$control_rt" "$method_rt"; do
  pid=$(cat "$root/wrapper.pid")
  printf '%s pid=%s alive=' "$root" "$pid"
  kill -0 "$pid" 2>/dev/null && echo yes || echo no
  test -f "$root/exit_code.txt" && { printf 'exit='; cat "$root/exit_code.txt"; } || true
done
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo MEMORY
cat /sys/fs/cgroup/memory.current
grep MemAvailable /proc/meminfo
cat /sys/fs/cgroup/memory.events
echo OUTPUTS
du -sh "$control_run" "$method_run" "$pair_rt"
echo METRIC_FILES
find "$control_run" "$method_run" -maxdepth 3 -type f \( -name 'metrics.log' -o -name 'events.out.tfevents*' \) -printf '%p %s\n' | sort
echo CHECKPOINTS
find "$control_run" "$method_run" -type d -path '*/checkpoints/global_step_*' -printf '%p\n' | sort -V | tail -n 16
echo CONTROL_STEPS
grep -E 'Global Step|global_step|evaluation|success' "$control_rt/foreground.log" | tail -n 35 || true
echo METHOD_STEPS
grep -E 'Global Step|global_step|evaluation|success|rlt_dvac' "$method_rt/foreground.log" | tail -n 45 || true
echo RESOURCE_TAIL
tail -n 8 "$pair_rt/paired_resources.csv" || true
echo FATAL_COUNTS
for root in "$control_rt" "$method_rt"; do
  printf '%s ' "$root"
  grep -E -c 'ActorDiedError|WorkerCrashedError|CUDA out of memory|OutOfMemoryError|SIGSEGV|FATAL' "$root/foreground.log" || true
done
