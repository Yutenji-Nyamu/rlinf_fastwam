#!/usr/bin/env bash
set -euo pipefail

OLD_ACTION=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
OLD_PRISM=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2
ACTION_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv-fix
ACTION_HEAD=e434f409b21d281ce883df29487ecae7cb3e4839
ST_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
ST_HEAD=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
ACTION_NAME=dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
ST_NAME=dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
ACTION_RUN="$ROOT/runs/$ACTION_NAME"
ST_RUN="$ROOT/runs/$ST_NAME"
ACTION_PACKET="$ROOT/packets/$ACTION_NAME"
ST_PACKET="$ROOT/packets/$ST_NAME"
ACTION_NAMESPACE=RLinf
PRISM_NAMESPACE=RLinf_1

test "$ACTION_HEAD" != __ACTION_FIX_HEAD__
test "$(git -C "$ACTION_WT" rev-parse HEAD)" = "$ACTION_HEAD"
test -z "$(git -C "$ACTION_WT" status --short)"
test "$(git -C "$ST_WT" rev-parse HEAD)" = "$ST_HEAD"
test -z "$(git -C "$ST_WT" status --short)"
for run in "$ACTION_RUN" "$ST_RUN"; do test ! -e "$run"; done
for packet in "$ACTION_PACKET" "$ST_PACKET"; do
  for file in resolved.yaml control_same_code_resolved.yaml parity.json contract.json command.txt packet_complete.txt; do
    test -s "$packet/$file"
  done
done
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

wrapper_contract() {
  local run=$1
  local pid pgid
  pid=$(<"$run/runtime/wrapper.pid")
  pgid=$(<"$run/runtime/owned.pgid")
  test "$pid" = "$pgid"
  kill -0 "$pid"
  test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
  ps -o args= -p "$pid" | grep -F "$run/runtime/wrapper.sh" >/dev/null
  printf '%s\n' "$pid"
}

gpu_unique_job() {
  local devices=$1 label=$2 minimum=$3
  local -a jobs=()
  local count=0 process_pid job
  while read -r process_pid; do
    [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
    test "$(ps -o user= -p "$process_pid" | xargs)" = chenyiteng
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    test -n "$job"
    jobs+=("$job")
    count=$((count + 1))
  done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  test "$count" -ge "$minimum"
  mapfile -t unique < <(printf '%s\n' "${jobs[@]}" | sort -u)
  test "${#unique[@]}" -eq 1
  printf '%s_gpu_processes=%s job=%s\n' "$label" "$count" "${unique[0]}" >&2
  printf '%s\n' "${unique[0]}"
}

namespace_count() {
  local namespace=$1
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$namespace" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_namespace_count", logging_level="ERROR")
print(sum(1 for row in ray.util.list_named_actors(all_namespaces=True)
          if row.get("namespace") == os.environ["TARGET_NAMESPACE"]))
ray.shutdown()
PY
}

wait_namespace_ready() {
  local namespace=$1 pid=$2 count=0
  for _ in $(seq 1 300); do
    kill -0 "$pid"
    count=$(namespace_count "$namespace")
    [[ "$count" == 15 ]] && break
    sleep 1
  done
  test "$count" = 15
}

stop_wrapper_and_observer() {
  local run=$1 pid=$2
  kill -TERM -- "-$pid"
  for _ in $(seq 1 60); do ! kill -0 "$pid" 2>/dev/null && break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid"; sleep 2; fi
  ! kill -0 "$pid" 2>/dev/null
  local observer
  observer=$(cat "$run/runtime/observer.pid" 2>/dev/null || true)
  if [[ -n "$observer" ]] && kill -0 "$observer" 2>/dev/null; then
    kill -TERM -- "-$observer" 2>/dev/null || kill -TERM "$observer" 2>/dev/null || true
  fi
}

cleanup_namespace_preserve_other() {
  local target=$1 preserve=$2
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$target" PRESERVE_NAMESPACE="$preserve" "$VENV/bin/python" - <<'PY'
import os, time, ray
target = os.environ["TARGET_NAMESPACE"]
preserve = os.environ["PRESERVE_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_dual_method_cutover", logging_level="ERROR")
def names(ns):
    return sorted(row["name"] for row in ray.util.list_named_actors(all_namespaces=True)
                  if row.get("namespace") == ns)
targets = names(target)
preserved = names(preserve)
assert len(targets) == 15, targets
assert len(preserved) == 15, preserved
managers = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
for name in sorted(targets, key=lambda item: (item in managers, item)):
    try:
        ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
    except ValueError:
        pass
for _ in range(120):
    remaining = names(target)
    if not remaining:
        break
    time.sleep(1)
else:
    raise SystemExit(f"namespace cleanup incomplete: {remaining}")
assert names(preserve) == preserved, f"preserved namespace changed: {preserve}"
print(f"target_namespace={target} actors_after=0 preserved_namespace={preserve} actors={len(preserved)}")
ray.shutdown()
PY
}

wait_old_job_gone() {
  local devices=$1 old_job=$2 remaining=0 process_pid job
  for _ in $(seq 1 120); do
    remaining=0
    while read -r process_pid; do
      [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
      job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
      [[ "$job" == "$old_job" ]] && remaining=$((remaining + 1))
    done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
    (( remaining == 0 )) && break
    sleep 1
  done
  test "$remaining" -eq 0
}

setup_source_env() {
  local wt=$1
  source "$VENV/bin/activate"
  unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
  export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$wt"
  export EMBODIED_PATH="$wt/examples/embodiment" RLINF_CODE_WORKING_DIR="$wt"
  export PYTHONPATH="$wt:$ROBOTWIN" OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
}

start_runtime() {
  local run=$1 packet=$2 head=$3 placement=$4 method=$5
  mkdir -p "$run/runtime"
  cp "$packet"/{resolved.yaml,control_same_code_resolved.yaml,parity.json,contract.json,command.txt,packet_complete.txt} "$run/runtime/"
  printf '%s\n' \
    "started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" \
    "source_head=$head" \
    "physical_gpus=$placement" \
    'fresh_start=true; target_step=100' \
    'train=64 env x 4 epochs = 256 trajectories/step; G8; max1024 records' \
    'actor=GB1024/MB32/update2' \
    "method=$method" \
    'eval=fixed32 every5; checkpoint=default DCP every10' \
    'normal_stop=complete step100/checkpoint/exit0; hard_timeout=216000s' \
    > "$run/runtime/launch_manifest.txt"

  cat > "$run/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 216000s "$@" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
WRAP
  chmod 700 "$run/runtime/wrapper.sh"
}

start_action_fix() {
  setup_source_env "$ACTION_WT"
  local -a args=(
    --config-path "$ACTION_WT/examples/embodiment/config" --config-name robotwin_adjust_bottle_grpo_openpi
    'cluster.component_placement={actor\, env\, rollout:"4,5"}'
    "runner.logger.log_path=$ACTION_RUN"
    runner.logger.experiment_name=robotwin_dvac_action_adv_fix_w0to2_formal100_2gpu64x4_b1024_fixed32_eval5_phys45_v1
    runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
    algorithm.update_epoch=2 algorithm.adv_type=grpo algorithm.filter_rewards=true algorithm.logprob_type=action_level
    algorithm.dvac_gradient_weighting.mode=apply algorithm.dvac_gradient_weighting.application=action_advantage
    algorithm.dvac_gradient_weighting.selected_l=3 algorithm.dvac_gradient_weighting.warmup_steps=1 algorithm.dvac_gradient_weighting.window_steps=5
    algorithm.dvac_gradient_weighting.weight_min=0.0 algorithm.dvac_gradient_weighting.weight_max=2.0
    env.train.total_num_envs=64 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true
    "env.train.video_cfg.video_base_dir=$ACTION_RUN/video/train" "env.train.task_config.save_path=$ACTION_RUN/robotwin_data/train"
    env.eval.total_num_envs=32 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true
    "env.eval.video_cfg.video_base_dir=$ACTION_RUN/video/eval" "env.eval.task_config.save_path=$ACTION_RUN/robotwin_data/eval"
    actor.micro_batch_size=32 actor.global_batch_size=1024 "actor.model.model_path=$MODEL"
  )
  start_runtime "$ACTION_RUN" "$ACTION_PACKET" "$ACTION_HEAD" 4,5 'Action-Adv Fix [0,2]: A_eff=A*w; action-level clip; sum valid H then mean queries'
  ACTION_START_EPOCH=$(date +%s)
  nohup setsid bash "$ACTION_RUN/runtime/wrapper.sh" "$ACTION_RUN/runtime" \
    "$VENV/bin/python" "$ACTION_WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" \
    > "$ACTION_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
  ACTION_PID=$!
  printf '%s\n' "$ACTION_PID" > "$ACTION_RUN/runtime/wrapper.pid"
  printf '%s\n' "$ACTION_PID" > "$ACTION_RUN/runtime/owned.pgid"
}

start_st_half() {
  setup_source_env "$ST_WT"
  local -a args=(
    --config-path "$ST_WT/examples/embodiment/config" --config-name robotwin_adjust_bottle_grpo_openpi
    'cluster.component_placement={actor\, env\, rollout:"6,7"}'
    "runner.logger.log_path=$ST_RUN"
    runner.logger.experiment_name=robotwin_dvac_st_global_z_w0p5to1p5_formal100_2gpu64x4_b1024_fixed32_eval5_phys67_v1
    runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
    algorithm.update_epoch=2 algorithm.adv_type=grpo algorithm.filter_rewards=true algorithm.logprob_type=chunk_level
    algorithm.dvac_gradient_weighting.mode=apply algorithm.dvac_gradient_weighting.selected_l=3
    algorithm.dvac_gradient_weighting.warmup_steps=1 algorithm.dvac_gradient_weighting.window_steps=5
    algorithm.dvac_gradient_weighting.weight_min=0.5 algorithm.dvac_gradient_weighting.weight_max=1.5
    env.train.total_num_envs=64 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true
    "env.train.video_cfg.video_base_dir=$ST_RUN/video/train" "env.train.task_config.save_path=$ST_RUN/robotwin_data/train"
    env.eval.total_num_envs=32 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true
    "env.eval.video_cfg.video_base_dir=$ST_RUN/video/eval" "env.eval.task_config.save_path=$ST_RUN/robotwin_data/eval"
    actor.micro_batch_size=32 actor.global_batch_size=1024 "actor.model.model_path=$MODEL"
  )
  start_runtime "$ST_RUN" "$ST_PACKET" "$ST_HEAD" 6,7 'ST-DVAC global-z [0.5,1.5]: Control joint-chunk forward/clip; local backward scaling'
  ST_START_EPOCH=$(date +%s)
  nohup setsid bash "$ST_RUN/runtime/wrapper.sh" "$ST_RUN/runtime" \
    "$VENV/bin/python" "$ST_WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" \
    > "$ST_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
  ST_PID=$!
  printf '%s\n' "$ST_PID" > "$ST_RUN/runtime/wrapper.pid"
  printf '%s\n' "$ST_PID" > "$ST_RUN/runtime/owned.pgid"
}

start_observer() {
  local run=$1 pid=$2 gpu_a=$3 gpu_b=$4
  cat > "$run/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2; gpu_a=$3; gpu_b=$4
printf '%s\n' "timestamp,driver_alive,host_mem_available_kib,gpu${gpu_a}_used_mib,gpu${gpu_a}_util_pct,gpu${gpu_b}_used_mib,gpu${gpu_b}_util_pct" > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i "$gpu_a,$gpu_b" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
  chmod 700 "$run/runtime/observer.sh"
  nohup setsid bash "$run/runtime/observer.sh" "$pid" "$run/runtime/resource.csv" "$gpu_a" "$gpu_b" > "$run/runtime/observer.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run/runtime/observer.pid"
}

# Dynamic old-job discovery prevents stale hard-coded Ray job IDs.
OLD_ACTION_PID=$(wrapper_contract "$OLD_ACTION")
OLD_PRISM_PID=$(wrapper_contract "$OLD_PRISM")
OLD_ACTION_JOB=$(gpu_unique_job 4,5 old_action 6)
OLD_PRISM_JOB=$(gpu_unique_job 6,7 old_prism 6)
test "$OLD_ACTION_JOB" != "$OLD_PRISM_JOB"
test "$(namespace_count "$ACTION_NAMESPACE")" = 15
test "$(namespace_count "$PRISM_NAMESPACE")" = 15

# Replace GPU4/5 first while preserving the still-running Prism namespace.
action_cutover_epoch=$(date +%s)
action_last_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$OLD_ACTION/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
stop_wrapper_and_observer "$OLD_ACTION" "$OLD_ACTION_PID"
cleanup_namespace_preserve_other "$ACTION_NAMESPACE" "$PRISM_NAMESPACE"
wait_old_job_gone 4,5 "$OLD_ACTION_JOB"
kill -0 "$OLD_PRISM_PID"

cat > "$OLD_ACTION/runtime/stopped_by_user_for_action_adv_fix.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested replacement by Action-Adv Fix [0,2]
last_complete_step=$action_last_step
ray_job_id=$OLD_ACTION_JOB
namespace=$ACTION_NAMESPACE
cleanup=exact owned PGID and exact namespace; Prism/shared Ray/other users unchanged
EOF

start_action_fix
start_observer "$ACTION_RUN" "$ACTION_PID" 4 5
wait_namespace_ready "$ACTION_NAMESPACE" "$ACTION_PID"
ACTION_NEW_JOB=$(gpu_unique_job 4,5 action_fix 6)
test "$ACTION_NEW_JOB" != "$OLD_ACTION_JOB"

# Replace GPU6/7 only after the new GPU4/5 namespace is complete.
prism_cutover_epoch=$(date +%s)
prism_last_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$OLD_PRISM/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
stop_wrapper_and_observer "$OLD_PRISM" "$OLD_PRISM_PID"
cleanup_namespace_preserve_other "$PRISM_NAMESPACE" "$ACTION_NAMESPACE"
wait_old_job_gone 6,7 "$OLD_PRISM_JOB"
kill -0 "$ACTION_PID"

cat > "$OLD_PRISM/runtime/stopped_by_user_for_st_dvac_half.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested replacement by ST-DVAC global-z [0.5,1.5]
last_complete_step=$prism_last_step
ray_job_id=$OLD_PRISM_JOB
namespace=$PRISM_NAMESPACE
cleanup=exact owned PGID and exact namespace; Action-Fix/shared Ray/other users unchanged
EOF

start_st_half
start_observer "$ST_RUN" "$ST_PID" 6 7
wait_namespace_ready "$PRISM_NAMESPACE" "$ST_PID"
ST_NEW_JOB=$(gpu_unique_job 6,7 st_half 6)
test "$ST_NEW_JOB" != "$OLD_PRISM_JOB"
test "$ACTION_NEW_JOB" != "$ST_NEW_JOB"
kill -0 "$ACTION_PID"
kill -0 "$ST_PID"

printf 'old_action_last_step=%s\nold_prism_last_step=%s\n' "$action_last_step" "$prism_last_step"
printf 'action_fix_run=%s\naction_fix_pid=%s\naction_fix_job=%s\naction_cutover_gap_seconds=%s\n' \
  "$ACTION_RUN" "$ACTION_PID" "$ACTION_NEW_JOB" "$(( ACTION_START_EPOCH-action_cutover_epoch ))"
printf 'st_half_run=%s\nst_half_pid=%s\nst_half_job=%s\nst_cutover_gap_seconds=%s\n' \
  "$ST_RUN" "$ST_PID" "$ST_NEW_JOB" "$(( ST_START_EPOCH-prism_cutover_epoch ))"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_ACTION_PRISM_TO_ACTION_FIX_ST_HALF_DUAL_FORMAL100_CUTOVER_OK
