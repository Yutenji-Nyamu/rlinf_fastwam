set -euo pipefail

control_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v2/runtime
method_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2/runtime
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v2

echo TIME
date -Is
echo LAUNCH
cat "$pair_runtime/launch_summary.txt"
echo WRAPPERS
for root in "$control_runtime" "$method_runtime"; do
  printf '%s ' "$root"
  pid=$(cat "$root/wrapper.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "alive pid=$pid"; else echo "not-alive pid=$pid"; fi
  if test -f "$root/compose_exit_code.txt"; then printf 'compose='; cat "$root/compose_exit_code.txt"; fi
  if test -f "$root/exit_code.txt"; then printf 'exit='; cat "$root/exit_code.txt"; fi
done
echo RAY
/root/autodl-tmp/RLinf/.venv/bin/ray status --address=172.17.0.9:52001 || true
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo MEMORY
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
grep MemAvailable /proc/meminfo
echo CONTROL_LOG_TAIL
tail -n 45 "$control_runtime/foreground.log" || true
echo METHOD_LOG_TAIL
tail -n 45 "$method_runtime/foreground.log" || true
echo FATAL_SCAN
grep -E -n 'Traceback|ModuleNotFoundError|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|ActorDiedError|FATAL' \
  "$control_runtime/foreground.log" "$method_runtime/foreground.log" || true
echo RESOLVED_CORE
for root in "$control_runtime" "$method_runtime"; do
  echo "--- $root"
  if test -f "$root/resolved.yaml"; then
    grep -E -n '^(  max_steps:|  val_check_interval:|  save_interval:|    total_num_envs:|    rollout_epoch:|  actor_global_batch_size:|  actor_micro_batch_size:|    max_updates_per_train_step:|    warmup_min_size:|    warmup_post_collect_updates:|    warmup_updates:|    ramp_updates:|    cache_size:|    sample_window_size:|    application:|    selected_l:|    z_clip:|    strength:|    success_scale:|      cuda_visible_devices:)' "$root/resolved.yaml" || true
  fi
done
echo RESOURCE_TAIL
tail -n 6 "$pair_runtime/paired_resources.csv" || true
