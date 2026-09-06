#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
HEAD=554c6dc8d586162d9444c01fa88308ed4f5203d0
BRANCH=codex/sz-7d07a421-grpo-pi0-robotwin

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$WT" status --porcelain)"
test "$(git -C "$WT" rev-parse "personal/$BRANCH")" = "$HEAD"
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test ! -e "$RUN"

if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 4-7 are not idle' >&2
  exit 1
fi
if pgrep -u "$(id -u)" -x raylet >/dev/null || pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
  printf '%s\n' 'chenyiteng already has a live Ray cluster' >&2
  exit 1
fi

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "$RUN"
ARGS=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_grpo_openpi
  'cluster.component_placement={actor\, env\, rollout:4-7}'
  "runner.logger.log_path=$RUN"
  'runner.max_epochs=1000'
  'runner.max_steps=100'
  'runner.val_check_interval=10'
  'runner.save_interval=10'
  'runner.resume_dir=null'
  'algorithm.update_epoch=2'
  'env.train.total_num_envs=128'
  'env.train.rollout_epoch=4'
  'env.train.max_episode_steps=200'
  'env.train.max_steps_per_rollout_epoch=200'
  "env.train.assets_path=$ROBOTWIN"
  'env.eval.total_num_envs=64'
  'env.eval.rollout_epoch=1'
  'env.eval.max_episode_steps=200'
  'env.eval.max_steps_per_rollout_epoch=200'
  'env.eval.use_fixed_reset_state_ids=true'
  "env.eval.assets_path=$ROBOTWIN"
  'actor.micro_batch_size=32'
  'actor.global_batch_size=2048'
  "actor.model.model_path=$MODEL"
)

"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${ARGS[@]}" --cfg job --resolve > "$RUN/resolved.yaml"
sha256sum "$RUN/resolved.yaml" > "$RUN/resolved.yaml.sha256"

cat > "$RUN/launch_manifest.txt" <<EOF
launch_time=$(date --iso-8601=seconds)
source_head=$HEAD
retry_of=grpo-formal100-current-4gpu128train64eval-ppo-matched-v1
retry_reason=Ray control-plane GCS RPC loss during step2 rollout3of4
training_config_change=none
diagnostic_change=driver exit record and final Ray control-plane log copy
physical_gpus=4,5,6,7
train_envs=128
train_rollout_epoch=4
group_size=8
train_trajectories_per_step=512
max_chunk_records_per_step=2048
global_batch=2048
micro_batch=32
update_epoch=2
optimizer_calls_per_step=2
eval_envs=64
val_interval=10
save_interval=10
max_steps=100
EOF

{
  printf '%s\n' '#!/usr/bin/env bash' 'set +e'
  declare -p VENV WT RUN ARGS
  printf '%s\n' '"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$RUN/driver.log" 2>&1'
  printf '%s\n' 'rc=$?'
  printf '%s\n' 'printf "exit_time=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$rc" > "$RUN/driver.exit"'
  printf '%s\n' 'exit "$rc"'
} > "$RUN/driver_wrapper.sh"
chmod 700 "$RUN/driver_wrapper.sh"

cat > "$RUN/resource_observer.sh" <<'OBSERVER'
#!/usr/bin/env bash
set -u
pid=$1
out=$2
run=$3
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,cgroup_memory_current_bytes,gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct,gpu6_used_mib,gpu6_util_pct,gpu7_used_mib,gpu7_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  ts=$(date --iso-8601=seconds)
  mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$pid/cgroup" 2>/dev/null)
  current=''
  test -n "$rel" && test -r "/sys/fs/cgroup$rel/memory.current" && current=$(cat "/sys/fs/cgroup$rel/memory.current")
  mapfile -t gpu < <(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,1,%s,%s' "$ts" "$mem" "$current" >> "$out"
  for row in "${gpu[@]}"; do printf ',%s' "$row" >> "$out"; done
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"

sleep 3
session=$(readlink -f /tmp/ray/session_latest 2>/dev/null || true)
printf 'ray_session=%s\ncopy_time=%s\n' "$session" "$(date --iso-8601=seconds)" > "$run/ray_log_snapshot.txt"
if test -d "$session/logs"; then
  mkdir -p "$run/ray_logs_final"
  find "$session/logs" -maxdepth 1 -type f \
    \( -name 'gcs_server.*' -o -name 'raylet.*' -o -name 'monitor.log' -o -name 'monitor.err' \) \
    -exec cp -p -t "$run/ray_logs_final" {} +
fi
OBSERVER
chmod 700 "$RUN/resource_observer.sh"

nohup setsid bash "$RUN/driver_wrapper.sh" > /dev/null 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/driver.pid"
nohup setsid bash "$RUN/resource_observer.sh" "$pid" "$RUN/resource.csv" "$RUN" \
  > "$RUN/resource_observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$RUN/resource_observer.pid"

sleep 10
kill -0 "$pid"
kill -0 "$observer_pid"
printf 'run=%s\ndriver_pid=%s\nobserver_pid=%s\nresolved_sha256=%s\ndata_avail_kib=%s\n' \
  "$RUN" "$pid" "$observer_pid" "$(awk '{print $1}' "$RUN/resolved.yaml.sha256")" \
  "$(df --output=avail -k /data | tail -n 1 | tr -d ' ')"
tail -n 20 "$RUN/driver.log" || true
printf '%s\n' 'SZ_GRPO_FORMAL100_RETRY_V2_LAUNCHED'
