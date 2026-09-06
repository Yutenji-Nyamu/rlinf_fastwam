set -euo pipefail
control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v2/runtime
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2/runtime
date -Is
for root in "$control" "$method"; do
  pid=$(cat "$root/wrapper.pid")
  printf '%s alive=' "$root"
  kill -0 "$pid" 2>/dev/null && echo yes || echo no
  test -f "$root/exit_code.txt" && { printf 'exit='; cat "$root/exit_code.txt"; } || true
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.events
echo CONTROL_PROGRESS
grep -E 'hardware ranks: \[\[0\]\]|Generating Rollout|Global Step|Actor train metrics' "$control/foreground.log" | tail -n 8 || true
echo METHOD_PROGRESS
grep -E 'hardware ranks: \[\[1\]\]|Generating Rollout|Global Step|Actor train metrics' "$method/foreground.log" | tail -n 8 || true
echo METHOD_CONFIG
grep -E -n '^(  max_steps:|  val_check_interval:|  save_interval:|    total_num_envs:|    rollout_epoch:|  actor_global_batch_size:|  actor_micro_batch_size:|    max_updates_per_train_step:|    warmup_min_size:|    warmup_post_collect_updates:|    warmup_updates:|    ramp_updates:|    cache_size:|    sample_window_size:|    application:|    selected_l:|    z_clip:|    strength:|    success_scale:)' "$method/resolved.yaml" || true
