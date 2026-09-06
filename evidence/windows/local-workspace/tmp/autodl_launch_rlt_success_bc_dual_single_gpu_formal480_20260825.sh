set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
python=/root/autodl-tmp/RLinf/.venv/bin/python
ray_cli=/root/autodl-tmp/RLinf/.venv/bin/ray
config_dir="$repo/examples/embodiment/config"
entry=examples/embodiment/train_embodied_agent.py
expected_head=64f2779f266b7d7019895c7aee1ebc222312b7d3

control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_formal480_20260825_v1
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v1
control_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v1/runtime
method_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v1/runtime
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v1

for target in "$control_run" "$method_run" "${control_runtime%/runtime}" "${method_runtime%/runtime}" "$pair_runtime"; do
  if test -e "$target"; then
    echo "refusing to reuse existing formal target: $target" >&2
    exit 20
  fi
done

actual_head=$(git -C "$repo" rev-parse HEAD)
test "$actual_head" = "$expected_head"
test -z "$(git -C "$repo" status --short)"

mkdir -p "$control_runtime" "$method_runtime" "$pair_runtime"

control_name=robotwin_adjust_bottle_rlt_single_gpu_control_formal480_20260825_v1
method_name=robotwin_adjust_bottle_rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v1
control_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control
method_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_gpu1_fresh480

control_cmd=(
  "$python" -B "$entry"
  --config-path "$config_dir"
  --config-name "$control_config"
  "runner.logger.log_path=$control_run"
  "runner.logger.experiment_name=$control_name"
  runner.max_steps=480
  runner.val_check_interval=25
  runner.save_interval=25
  runner.resume_dir=null
  "env.train.task_config.save_path=$control_run/robotwin_data/train"
  "env.eval.task_config.save_path=$control_run/robotwin_data/eval"
)

method_cmd=(
  "$python" -B "$entry"
  --config-path "$config_dir"
  --config-name "$method_config"
  "runner.logger.log_path=$method_run"
  "runner.logger.experiment_name=$method_name"
  runner.max_steps=480
  runner.val_check_interval=25
  runner.save_interval=25
  runner.resume_dir=null
  "env.train.task_config.save_path=$method_run/robotwin_data/train"
  "env.eval.task_config.save_path=$method_run/robotwin_data/eval"
)

printf '%q ' "${control_cmd[@]}" > "$control_runtime/exact_command.txt"
printf '\n' >> "$control_runtime/exact_command.txt"
printf '%q ' "${method_cmd[@]}" > "$method_runtime/exact_command.txt"
printf '\n' >> "$method_runtime/exact_command.txt"

compose_env=(
  env
  "REPO_PATH=$repo"
  "RLINF_CODE_WORKING_DIR=$repo"
  "PYTHONPATH=$repo"
  HYDRA_FULL_ERROR=1
)

(
  cd "$repo"
  "${compose_env[@]}" "${control_cmd[@]}" --cfg job --resolve
) > "$control_runtime/resolved.yaml" 2> "$control_runtime/compose.stderr.log"
(
  cd "$repo"
  "${compose_env[@]}" "${method_cmd[@]}" --cfg job --resolve
) > "$method_runtime/resolved.yaml" 2> "$method_runtime/compose.stderr.log"

sha256sum "$control_runtime/resolved.yaml" > "$control_runtime/resolved.sha256"
sha256sum "$method_runtime/resolved.yaml" > "$method_runtime/resolved.sha256"
printf '%s\n' "$actual_head" > "$control_runtime/source_head.txt"
printf '%s\n' "$actual_head" > "$method_runtime/source_head.txt"
sha256sum \
  "$repo/rlinf/envs/robotwin/seeds/train_seeds.json" \
  "$repo/rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json" \
  > "$pair_runtime/seed_files.sha256"

grep -E '^(  max_steps:|  val_check_interval:|  save_interval:|    total_num_envs:|    rollout_epoch:|  actor_global_batch_size:|  actor_micro_batch_size:|    max_updates_per_train_step:|    warmup_min_size:|    warmup_post_collect_updates:|    warmup_updates:|    ramp_updates:|    cache_size:|    sample_window_size:|    application:|    selected_l:|    z_clip:|    strength:|    success_scale:)' \
  "$control_runtime/resolved.yaml" > "$control_runtime/formal_contract_excerpt.txt" || true
grep -E '^(  max_steps:|  val_check_interval:|  save_interval:|    total_num_envs:|    rollout_epoch:|  actor_global_batch_size:|  actor_micro_batch_size:|    max_updates_per_train_step:|    warmup_min_size:|    warmup_post_collect_updates:|    warmup_updates:|    ramp_updates:|    cache_size:|    sample_window_size:|    application:|    selected_l:|    z_clip:|    strength:|    success_scale:)' \
  "$method_runtime/resolved.yaml" > "$method_runtime/formal_contract_excerpt.txt" || true

node_ip=$(hostname -I | awk '{print $1}')
ray_port=50011
ray_address="$node_ip:$ray_port"
ray_temp=/tmp/rltbc_f480_50011
if ss -ltn | awk '{print $4}' | grep -q ":$ray_port$"; then
  echo "Ray port already in use: $ray_port" >&2
  exit 21
fi

setsid env RAY_USAGE_STATS_ENABLED=0 "$ray_cli" start \
  --head \
  --node-ip-address="$node_ip" \
  --port="$ray_port" \
  --num-cpus=36 \
  --num-gpus=2 \
  --include-dashboard=false \
  --disable-usage-stats \
  --temp-dir="$ray_temp" \
  --block \
  > "$pair_runtime/ray_head.log" 2>&1 < /dev/null &
ray_head_pid=$!
printf '%s\n' "$ray_head_pid" > "$pair_runtime/ray_head.pid"
printf '%s\n' "$ray_address" > "$pair_runtime/ray_address.txt"
printf '%s\n' "$ray_temp" > "$pair_runtime/ray_temp_dir.txt"

ray_ready=0
for _ in $(seq 1 30); do
  if "$ray_cli" status --address="$ray_address" > "$pair_runtime/ray_status.txt" 2>&1; then
    ray_ready=1
    break
  fi
  sleep 2
done
test "$ray_ready" = 1

control_line=$(printf '%q ' "${control_cmd[@]}")
method_line=$(printf '%q ' "${method_cmd[@]}")
common_exports="export RAY_ADDRESS=$(printf '%q' "$ray_address") REPO_PATH=$(printf '%q' "$repo") RLINF_CODE_WORKING_DIR=$(printf '%q' "$repo") PYTHONPATH=$(printf '%q' "$repo") TORCHINDUCTOR_COMPILE_THREADS=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0"

control_runner="cd $(printf '%q' "$repo"); $common_exports; date -Is > $(printf '%q' "$control_runtime/started_at.txt"); $control_line; rc=\$?; printf '%s\\n' \"\$rc\" > $(printf '%q' "$control_runtime/exit_code.txt"); date -Is > $(printf '%q' "$control_runtime/finished_at.txt"); exit \$rc"
setsid bash -lc "$control_runner" > "$control_runtime/foreground.log" 2>&1 < /dev/null &
control_pid=$!
printf '%s\n' "$control_pid" > "$control_runtime/wrapper.pid"
printf '0\n' > "$control_runtime/expected_physical_gpu.txt"

method_runner="sleep 120; cd $(printf '%q' "$repo"); $common_exports; date -Is > $(printf '%q' "$method_runtime/started_at.txt"); $method_line; rc=\$?; printf '%s\\n' \"\$rc\" > $(printf '%q' "$method_runtime/exit_code.txt"); date -Is > $(printf '%q' "$method_runtime/finished_at.txt"); exit \$rc"
setsid bash -lc "$method_runner" > "$method_runtime/foreground.log" 2>&1 < /dev/null &
method_pid=$!
printf '%s\n' "$method_pid" > "$method_runtime/wrapper.pid"
printf '1\n' > "$method_runtime/expected_physical_gpu.txt"

monitor_line="printf 'timestamp,memory_current,memory_available_kb,gpu0_mib,gpu0_util,gpu1_mib,gpu1_util\\n' > $(printf '%q' "$pair_runtime/paired_resources.csv"); while kill -0 $control_pid 2>/dev/null || kill -0 $method_pid 2>/dev/null; do ts=\$(date -Is); mc=\$(cat /sys/fs/cgroup/memory.current); ma=\$(awk '/MemAvailable/ {print \$2}' /proc/meminfo); g=\$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\\n' ',' | sed 's/,$//'); printf '%s,%s,%s,%s\\n' \"\$ts\" \"\$mc\" \"\$ma\" \"\$g\" >> $(printf '%q' "$pair_runtime/paired_resources.csv"); sleep 5; done"
setsid bash -lc "$monitor_line" > "$pair_runtime/monitor.log" 2>&1 < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" > "$pair_runtime/monitor.pid"

cleanup_line="while kill -0 $control_pid 2>/dev/null || kill -0 $method_pid 2>/dev/null; do sleep 60; done; kill -INT -- -$ray_head_pid 2>/dev/null || true; sleep 10; kill -TERM -- -$ray_head_pid 2>/dev/null || true; date -Is > $(printf '%q' "$pair_runtime/cleanup_finished_at.txt")"
setsid bash -lc "$cleanup_line" > "$pair_runtime/cleanup.log" 2>&1 < /dev/null &
cleanup_pid=$!
printf '%s\n' "$cleanup_pid" > "$pair_runtime/cleanup.pid"

{
  printf 'source_head=%s\n' "$actual_head"
  printf 'shared_ray_address=%s\n' "$ray_address"
  printf 'shared_ray_head_pid=%s\n' "$ray_head_pid"
  printf 'control_gpu=0\n'
  printf 'control_pid=%s\n' "$control_pid"
  printf 'method_gpu=1\n'
  printf 'method_delayed_pid=%s\n' "$method_pid"
  printf 'method_stagger_seconds=120\n'
  printf 'monitor_pid=%s\n' "$monitor_pid"
  printf 'cleanup_pid=%s\n' "$cleanup_pid"
  printf 'control_run=%s\n' "$control_run"
  printf 'method_run=%s\n' "$method_run"
} > "$pair_runtime/launch_summary.txt"

echo 'FORMAL_PAIR_LAUNCHED'
cat "$pair_runtime/launch_summary.txt"
echo 'CONTROL_CONTRACT'
cat "$control_runtime/formal_contract_excerpt.txt"
echo 'METHOD_CONTRACT'
cat "$method_runtime/formal_contract_excerpt.txt"
