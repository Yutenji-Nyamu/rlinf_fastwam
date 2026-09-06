#!/usr/bin/env bash
set -euo pipefail
control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_matched_width_formal480_20260826_v3/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_matched_width_formal480_20260826_v3

date -Is
for label in control method; do
  if test "$label" = control; then rt=$control_rt; else rt=$method_rt; fi
  pid=$(cat "$rt/wrapper.pid")
  printf '%s pid=%s alive=' "$label" "$pid"
  kill -0 "$pid" 2>/dev/null && echo yes || echo no
  test -f "$rt/exit_code.txt" && { printf '%s exit=' "$label"; cat "$rt/exit_code.txt"; } || true
  printf '%s compose=' "$label"; cat "$rt/compose_exit_code.txt" 2>/dev/null || echo pending
  if test -f "$rt/resolved.yaml"; then
    grep -E '^  micro_batch_size:|^    warmup_min_size:|^    cache_size:|^    sample_window_size:|^  max_steps:|^    mode:|^    application:|^    strength:' "$rt/resolved.yaml"
  fi
  echo "$label tail"
  tail -n 20 "$rt/foreground.log" 2>/dev/null || true
done
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo MEMORY
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
echo RAY
/root/autodl-tmp/RLinf/.venv/bin/ray status --address="$(cat "$pair_rt/ray_address.txt")" | sed -n '1,55p'
echo FATAL
for rt in "$control_rt" "$method_rt"; do
  grep -E -c 'ActorDiedError|WorkerCrashedError|CUDA out of memory|OutOfMemoryError|SIGSEGV|FATAL|Traceback' "$rt/foreground.log" 2>/dev/null || true
done
