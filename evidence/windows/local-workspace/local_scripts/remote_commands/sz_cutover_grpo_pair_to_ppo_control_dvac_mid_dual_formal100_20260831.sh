#!/usr/bin/env bash
set -euo pipefail

OLD_ACTION=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
OLD_ST=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=74617ced87d64045ab6850d0efd90956a494af66
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo
CONTROL_NAME=ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1
DVAC_NAME=ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1
CONTROL_RUN="$ROOT/runs/$CONTROL_NAME"
DVAC_RUN="$ROOT/runs/$DVAC_NAME"
CONTROL_PACKET="$ROOT/packets/$CONTROL_NAME"
DVAC_PACKET="$ROOT/packets/$DVAC_NAME"
CONTROL_EXPERIMENT=robotwin_ppo_control_formal100_2gpu64x4_b1024_fixed32_eval5_phys45_localshard_v1
DVAC_EXPERIMENT=robotwin_ppo_dvac_action_adv_fix_w0p5to1p5_formal100_2gpu64x4_b1024_fixed32_eval5_phys67_localshard_v1

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
for packet in "$CONTROL_PACKET" "$DVAC_PACKET"; do
  test -s "$packet/resolved.yaml"; test -s "$packet/contract.json"; test -s "$packet/command.txt"; test -s "$packet/packet_complete.txt"
  "$VENV/bin/python" - "$packet/contract.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); assert d["unexpected"] == []; assert d["common"]["checkpoint_format"] == "local_shard"
PY
done
test ! -e "$CONTROL_RUN"; test ! -e "$DVAC_RUN"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

wrapper_contract() {
  local run=$1 pid pgid
  pid=$(<"$run/runtime/wrapper.pid"); pgid=$(<"$run/runtime/owned.pgid")
  test "$pid" = "$pgid"; kill -0 "$pid"
  test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
  ps -o args= -p "$pid" | grep -F "$run/runtime/wrapper.sh" >/dev/null
  printf '%s\n' "$pid"
}

gpu_unique_job() {
  local devices=$1 label=$2 minimum=$3 count=0 process_pid job
  local -a jobs=()
  while read -r process_pid; do
    [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
    test "$(ps -o user= -p "$process_pid" | xargs)" = chenyiteng
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    test -n "$job"; jobs+=("$job"); count=$((count + 1))
  done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  test "$count" -ge "$minimum"
  mapfile -t unique < <(printf '%s\n' "${jobs[@]}" | sort -u)
  test "${#unique[@]}" -eq 1
  printf '%s_gpu_processes=%s job=%s\n' "$label" "$count" "${unique[0]}" >&2
  printf '%s\n' "${unique[0]}"
}

namespace_count() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_ppo_cutover_count", logging_level="ERROR")
print(sum(1 for row in ray.util.list_named_actors(all_namespaces=True) if row.get("namespace") == os.environ["TARGET_NAMESPACE"]))
ray.shutdown()
PY
}

stop_wrapper_and_observer() {
  local run=$1 pid=$2 observer
  kill -TERM -- "-$pid"
  for _ in $(seq 1 60); do ! kill -0 "$pid" 2>/dev/null && break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid"; sleep 2; fi
  ! kill -0 "$pid" 2>/dev/null
  observer=$(cat "$run/runtime/observer.pid" 2>/dev/null || true)
  if [[ -n "$observer" ]] && kill -0 "$observer" 2>/dev/null; then kill -TERM -- "-$observer" 2>/dev/null || kill -TERM "$observer" 2>/dev/null || true; fi
}

cleanup_namespace_preserve_other() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 PRESERVE_NAMESPACE=$2 PRESERVE_EXPECTED=$3 "$VENV/bin/python" - <<'PY'
import os, time, ray
target=os.environ["TARGET_NAMESPACE"]; preserve=os.environ["PRESERVE_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_ppo_exact_cleanup", logging_level="ERROR")
def names(ns): return sorted(r["name"] for r in ray.util.list_named_actors(all_namespaces=True) if r.get("namespace") == ns)
targets=names(target); preserved=names(preserve)
assert len(targets)==15, (target, targets)
assert len(preserved)==int(os.environ["PRESERVE_EXPECTED"]), (preserve, preserved)
managers={"CollectiveManager","DeviceLockManager","NodeManager","PortLockManager","WorkerManager"}
for name in sorted(targets, key=lambda x:(x in managers,x)):
    try: ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
    except ValueError: pass
for _ in range(120):
    if not names(target): break
    time.sleep(1)
else: raise RuntimeError(names(target))
assert names(preserve)==preserved
print(f"cleaned={target}; preserved={preserve}:{len(preserved)}")
ray.shutdown()
PY
}

wait_old_job_gone() {
  local devices=$1 old_job=$2 remaining process_pid job
  for _ in $(seq 1 120); do
    remaining=0
    while read -r process_pid; do
      [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
      job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
      [[ "$job" == "$old_job" ]] && remaining=$((remaining + 1))
    done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
    (( remaining == 0 )) && break; sleep 1
  done
  test "$remaining" -eq 0
}

wait_namespace_ready() {
  local namespace=$1 pid=$2 count=0
  for _ in $(seq 1 360); do
    kill -0 "$pid"; count=$(namespace_count "$namespace")
    [[ "$count" == 15 ]] && break; sleep 1
  done
  test "$count" = 15
}

setup_env() {
  source "$VENV/bin/activate"
  unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
  export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
  export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
  export PYTHONPATH="$WT:$ROBOTWIN" OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
}

start_runtime() {
  local run=$1 packet=$2 placement=$3 method=$4
  mkdir -p "$run/runtime"
  cp "$packet"/{resolved.yaml,contract.json,command.txt,packet_complete.txt,source_head.txt} "$run/runtime/"
  printf '%s\n' "started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" "source_head=$HEAD" "physical_gpus=$placement" \
    'fresh_start=true; target_step=100' 'train=64 env x 4 rollout epochs = 256 trajectories/step; max1024 query records' \
    'actor=GB1024/MB32/update2; PPO GAE+critic/group1' "method=$method" \
    'eval=fixed32 every5; videos=true; checkpoint=local_shard every10' \
    'normal_stop=complete step100/checkpoint/exit0; hard_timeout=216000s' > "$run/runtime/launch_manifest.txt"
  cat > "$run/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 216000s "$@" > "$runtime/driver.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$runtime/exit_code.txt"; date --iso-8601=seconds > "$runtime/finished_at.txt"; exit "$rc"
WRAP
  chmod 700 "$run/runtime/wrapper.sh"
}

start_observer() {
  local run=$1 pid=$2 gpu_a=$3 gpu_b=$4
  cat > "$run/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2; a=$3; b=$4
printf '%s\n' "timestamp,driver_alive,host_mem_available_kib,gpu${a}_used_mib,gpu${a}_util_pct,gpu${b}_used_mib,gpu${b}_util_pct" > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i "$a,$b" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"; sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
  chmod 700 "$run/runtime/observer.sh"
  nohup setsid bash "$run/runtime/observer.sh" "$pid" "$run/runtime/resource.csv" "$gpu_a" "$gpu_b" > "$run/runtime/observer.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run/runtime/observer.pid"
}

common=(
  --config-path "$WT/examples/embodiment/config" --config-name robotwin_adjust_bottle_ppo_openpi_dvac_action_adv
  runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=2 algorithm.adv_type=gae algorithm.loss_type=actor_critic algorithm.filter_rewards=false
  env.train.total_num_envs=64 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true
  actor.micro_batch_size=32 actor.global_batch_size=1024 "actor.model.model_path=$MODEL" actor.fsdp_config.checkpoint_format=local_shard
)
control=(
  'cluster.component_placement={actor\, env\, rollout:"4,5"}' "runner.logger.log_path=$CONTROL_RUN" "runner.logger.experiment_name=$CONTROL_EXPERIMENT"
  algorithm.logprob_type=chunk_level algorithm.dvac_gradient_weighting.mode=off algorithm.dvac_gradient_weighting.application=logprob_st
  algorithm.dvac_gradient_weighting.weight_min=null algorithm.dvac_gradient_weighting.weight_max=null
  "env.train.video_cfg.video_base_dir=$CONTROL_RUN/video/train" "env.train.task_config.save_path=$CONTROL_RUN/robotwin_data/train"
  "env.eval.video_cfg.video_base_dir=$CONTROL_RUN/video/eval" "env.eval.task_config.save_path=$CONTROL_RUN/robotwin_data/eval"
)
dvac=(
  'cluster.component_placement={actor\, env\, rollout:"6,7"}' "runner.logger.log_path=$DVAC_RUN" "runner.logger.experiment_name=$DVAC_EXPERIMENT"
  algorithm.logprob_type=action_level algorithm.dvac_gradient_weighting.mode=apply algorithm.dvac_gradient_weighting.application=action_advantage
  algorithm.dvac_gradient_weighting.selected_l=3 algorithm.dvac_gradient_weighting.warmup_steps=1 algorithm.dvac_gradient_weighting.window_steps=5
  algorithm.dvac_gradient_weighting.weight_min=0.5 algorithm.dvac_gradient_weighting.weight_max=1.5
  "env.train.video_cfg.video_base_dir=$DVAC_RUN/video/train" "env.train.task_config.save_path=$DVAC_RUN/robotwin_data/train"
  "env.eval.video_cfg.video_base_dir=$DVAC_RUN/video/eval" "env.eval.task_config.save_path=$DVAC_RUN/robotwin_data/eval"
)

old_action_pid=$(wrapper_contract "$OLD_ACTION"); old_st_pid=$(wrapper_contract "$OLD_ST")
old_action_job=$(gpu_unique_job 4,5 old_action 6); old_st_job=$(gpu_unique_job 6,7 old_st 6)
test "$old_action_job" != "$old_st_job"
test "$(namespace_count RLinf)" = 15; test "$(namespace_count RLinf_1)" = 15
old_action_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$OLD_ACTION/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
old_st_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$OLD_ST/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)

cutover_epoch=$(date +%s)
stop_wrapper_and_observer "$OLD_ACTION" "$old_action_pid"
stop_wrapper_and_observer "$OLD_ST" "$old_st_pid"
cleanup_namespace_preserve_other RLinf RLinf_1 15
cleanup_namespace_preserve_other RLinf_1 RLinf 0
wait_old_job_gone 4,5 "$old_action_job"; wait_old_job_gone 6,7 "$old_st_job"

cat > "$OLD_ACTION/runtime/stopped_by_user_for_ppo_pair_20260831.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested replacement by two-GPU PPO Control
last_complete_step=$old_action_step
ray_job_id=$old_action_job
namespace=RLinf
cleanup=owned PGID and exact namespace only; shared Ray and other users unchanged
EOF
cat > "$OLD_ST/runtime/stopped_by_user_for_ppo_pair_20260831.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested replacement by two-GPU PPO-DVAC Action-Adv Fix [0.5,1.5]
last_complete_step=$old_st_step
ray_job_id=$old_st_job
namespace=RLinf_1
cleanup=owned PGID and exact namespace only; shared Ray and other users unchanged
EOF

setup_env
start_runtime "$CONTROL_RUN" "$CONTROL_PACKET" 4,5 'PPO Control: GAE+value head; chunk-level clip; DVAC off'
control_start_epoch=$(date +%s)
nohup setsid bash "$CONTROL_RUN/runtime/wrapper.sh" "$CONTROL_RUN/runtime" "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${control[@]}" > "$CONTROL_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
control_pid=$!; printf '%s\n' "$control_pid" > "$CONTROL_RUN/runtime/wrapper.pid"; printf '%s\n' "$control_pid" > "$CONTROL_RUN/runtime/owned.pgid"
start_observer "$CONTROL_RUN" "$control_pid" 4 5
wait_namespace_ready RLinf "$control_pid"
control_job=$(gpu_unique_job 4,5 ppo_control 6)

test "$(namespace_count RLinf_1)" = 0
start_runtime "$DVAC_RUN" "$DVAC_PACKET" 6,7 'PPO-DVAC Action-Adv Fix [0.5,1.5]: PPO GAE actor advantage only; critic unchanged'
dvac_start_epoch=$(date +%s)
nohup setsid bash "$DVAC_RUN/runtime/wrapper.sh" "$DVAC_RUN/runtime" "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${dvac[@]}" > "$DVAC_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
dvac_pid=$!; printf '%s\n' "$dvac_pid" > "$DVAC_RUN/runtime/wrapper.pid"; printf '%s\n' "$dvac_pid" > "$DVAC_RUN/runtime/owned.pgid"
start_observer "$DVAC_RUN" "$dvac_pid" 6 7
wait_namespace_ready RLinf_1 "$dvac_pid"
dvac_job=$(gpu_unique_job 6,7 ppo_dvac 6)

test "$control_job" != "$dvac_job"; kill -0 "$control_pid"; kill -0 "$dvac_pid"
printf 'old_action_step=%s old_st_step=%s\n' "$old_action_step" "$old_st_step"
printf 'control_run=%s\ncontrol_pid=%s\ncontrol_job=%s\ncontrol_gap_seconds=%s\n' "$CONTROL_RUN" "$control_pid" "$control_job" "$((control_start_epoch-cutover_epoch))"
printf 'dvac_run=%s\ndvac_pid=%s\ndvac_job=%s\ndvac_gap_seconds=%s\n' "$DVAC_RUN" "$dvac_pid" "$dvac_job" "$((dvac_start_epoch-cutover_epoch))"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_GRPO_PAIR_TO_PPO_CONTROL_DVAC_MID_CUTOVER_OK
