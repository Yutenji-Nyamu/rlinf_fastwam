#!/usr/bin/env bash
set -euo pipefail

OLD_ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo/runs
OLD_CONTROL="$OLD_ROOT/ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1"
OLD_DVAC="$OLD_ROOT/ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1"
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed
HEAD=256eeeb4459b4bd5db85bfc6a0eb315771e8c38c
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
CONTROL_NAME=pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1
DVAC_NAME=pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1
CONTROL_RUN="$ROOT/runs/$CONTROL_NAME"
DVAC_RUN="$ROOT/runs/$DVAC_NAME"
CONTROL_PACKET="$ROOT/packets/$CONTROL_NAME"
DVAC_PACKET="$ROOT/packets/$DVAC_NAME"
CONTROL_EXPERIMENT=robotwin_pi05_grpo_control_formal100_2gpu64x4_g8_b512_u5_m5_fixed32_eval5_phys45_localshard_v1
DVAC_EXPERIMENT=robotwin_pi05_grpo_dvac_action_adv_w0p5to1p5_formal100_2gpu64x4_g8_b512_u5_m5_fixed32_eval5_phys67_localshard_v1

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
for packet in "$CONTROL_PACKET" "$DVAC_PACKET"; do
  test -s "$packet/resolved.yaml"; test -s "$packet/contract.json"
  test -s "$packet/command.txt"; test -s "$packet/source_head.txt"; test -s "$packet/packet_complete.txt"
  "$VENV/bin/python" - "$packet/contract.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["unexpected"] == []
assert d["common"]["checkpoint_format"] == "local_shard"
assert d["common"]["optimizer_steps_per_outer"] == 10
PY
done
test ! -e "$CONTROL_RUN"; test ! -e "$DVAC_RUN"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

namespace_count() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_pi05_cutover_count", logging_level="ERROR")
print(sum(1 for row in ray.util.list_named_actors(all_namespaces=True) if row.get("namespace") == os.environ["TARGET_NAMESPACE"]))
ray.shutdown()
PY
}

namespace_names() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_pi05_cutover_names", logging_level="ERROR")
print("\n".join(sorted(row["name"] for row in ray.util.list_named_actors(all_namespaces=True) if row.get("namespace") == os.environ["TARGET_NAMESPACE"])))
ray.shutdown()
PY
}

cleanup_namespace() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE=$1 "$VENV/bin/python" - <<'PY'
import os, time, ray
target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_pi05_exact_cleanup", logging_level="ERROR")
def rows(): return sorted((r["name"] for r in ray.util.list_named_actors(all_namespaces=True) if r.get("namespace") == target))
names = rows()
if names:
    assert len(names) == 15, (target, names)
    managers = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
    for name in sorted(names, key=lambda value: (value in managers, value)):
        try: ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
        except ValueError: pass
    for _ in range(120):
        if not rows(): break
        time.sleep(1)
    else: raise RuntimeError(rows())
print(f"cleaned_namespace={target}; actor_count={len(names)}")
ray.shutdown()
PY
}

wrapper_pid_exact() {
  local run=$1 pid
  pid=$(<"$run/runtime/wrapper.pid")
  test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
  ps -o args= -p "$pid" | grep -F "$run/runtime/wrapper.sh" >/dev/null
  printf '%s\n' "$pid"
}

gpu_unique_job() {
  local devices=$1 label=$2 minimum=$3 pid job count=0
  local -a jobs=()
  while read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/environ" ]] || continue
    test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
    job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    test -n "$job"; jobs+=("$job"); count=$((count + 1))
  done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  test "$count" -ge "$minimum"
  mapfile -t unique < <(printf '%s\n' "${jobs[@]}" | sort -u)
  test "${#unique[@]}" -eq 1
  printf '%s_gpu_processes=%s job=%s\n' "$label" "$count" "${unique[0]}" >&2
  printf '%s\n' "${unique[0]}"
}

stop_owned_group() {
  local run=$1 pid=$2 observer
  kill -TERM -- "-$pid"
  for _ in $(seq 1 90); do ! kill -0 "$pid" 2>/dev/null && break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid"; sleep 2; fi
  ! kill -0 "$pid" 2>/dev/null
  observer=$(cat "$run/runtime/observer.pid" 2>/dev/null || true)
  if [[ -n "$observer" ]] && kill -0 "$observer" 2>/dev/null; then
    kill -TERM -- "-$observer" 2>/dev/null || kill -TERM "$observer" 2>/dev/null || true
  fi
}

wait_job_gone() {
  local devices=$1 old_job=$2 pid job remaining
  for _ in $(seq 1 120); do
    remaining=0
    while read -r pid; do
      [[ -n "$pid" && -r "/proc/$pid/environ" ]] || continue
      job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
      [[ "$job" == "$old_job" ]] && remaining=$((remaining + 1))
    done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
    (( remaining == 0 )) && break
    sleep 1
  done
  test "$remaining" -eq 0
}

wait_namespace_ready() {
  local namespace=$1 pid=$2 count=0
  for _ in $(seq 1 360); do
    kill -0 "$pid"
    count=$(namespace_count "$namespace")
    [[ "$count" == 15 ]] && break
    sleep 1
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
  printf '%s\n' \
    "started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" "source_head=$HEAD" "physical_gpus=$placement" \
    'fresh_start=true; target_step=100' 'train=64 env x 4 rollout epochs = 256 trajectories/step; G8; max1024 query records' \
    'actor=GB512/MB32/update5; 10 optimizer calls/outer; pi0.5 M5' "method=$method" \
    'eval=fixed32 every5; videos=true; checkpoint=local_shard every10' \
    'normal_stop=complete step100/checkpoint/exit0; hard_timeout=259200s' > "$run/runtime/launch_manifest.txt"
  cat > "$run/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 259200s "$@" > "$runtime/driver.log" 2>&1
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
  --config-path "$WT/examples/embodiment/config" --config-name robotwin_adjust_bottle_grpo_openpi_pi05
  runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=5 algorithm.group_size=8 algorithm.adv_type=grpo algorithm.loss_type=actor algorithm.filter_rewards=true
  env.train.total_num_envs=64 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true
  actor.micro_batch_size=32 actor.global_batch_size=512 "actor.model.model_path=$MODEL" actor.model.num_steps=5
  ++actor.fsdp_config.checkpoint_format=local_shard
)
control=(
  'cluster.component_placement={actor\, env\, rollout:"4,5"}'
  "runner.logger.log_path=$CONTROL_RUN" "runner.logger.experiment_name=$CONTROL_EXPERIMENT"
  algorithm.logprob_type=chunk_level algorithm.dvac_gradient_weighting.mode=off
  algorithm.dvac_gradient_weighting.application=logprob_st
  algorithm.dvac_gradient_weighting.weight_min=null algorithm.dvac_gradient_weighting.weight_max=null
  "env.train.video_cfg.video_base_dir=$CONTROL_RUN/video/train" "env.train.task_config.save_path=$CONTROL_RUN/robotwin_data/train"
  "env.eval.video_cfg.video_base_dir=$CONTROL_RUN/video/eval" "env.eval.task_config.save_path=$CONTROL_RUN/robotwin_data/eval"
)
dvac=(
  'cluster.component_placement={actor\, env\, rollout:"6,7"}'
  "runner.logger.log_path=$DVAC_RUN" "runner.logger.experiment_name=$DVAC_EXPERIMENT"
  algorithm.logprob_type=action_level algorithm.dvac_gradient_weighting.mode=apply
  algorithm.dvac_gradient_weighting.application=action_advantage
  algorithm.dvac_gradient_weighting.selected_l=3 algorithm.dvac_gradient_weighting.warmup_steps=1
  algorithm.dvac_gradient_weighting.window_steps=5
  algorithm.dvac_gradient_weighting.weight_min=0.5 algorithm.dvac_gradient_weighting.weight_max=1.5
  "env.train.video_cfg.video_base_dir=$DVAC_RUN/video/train" "env.train.task_config.save_path=$DVAC_RUN/robotwin_data/train"
  "env.eval.video_cfg.video_base_dir=$DVAC_RUN/video/eval" "env.eval.task_config.save_path=$DVAC_RUN/robotwin_data/eval"
)

# Exact old-state contract: Method already exited from the Ray 95% memory monitor;
# Control alone remains in namespace RLinf on physical GPUs 4/5.
old_control_pid=$(wrapper_pid_exact "$OLD_CONTROL")
test ! -d "/proc/$(<"$OLD_DVAC/runtime/wrapper.pid")"
test "$(<"$OLD_DVAC/runtime/exit_code.txt")" = 255
test "$(namespace_count RLinf)" = 15
test "$(namespace_count RLinf_1)" = 0
old_control_job=$(gpu_unique_job 4,5 old_ppo_control 6)
test -z "$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
old_control_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$OLD_CONTROL/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
old_dvac_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$OLD_DVAC/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)

cutover_epoch=$(date +%s)
stop_owned_group "$OLD_CONTROL" "$old_control_pid"
cleanup_namespace RLinf
wait_job_gone 4,5 "$old_control_job"

cat > "$OLD_CONTROL/runtime/stopped_by_user_for_pi05_grpo_pair_20260901.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested replacement by pi0.5 GRPO Control
last_complete_step=$old_control_step
ray_job_id=$old_control_job
namespace=RLinf
cleanup=owned PGID and exact namespace only; shared Ray and other users unchanged
EOF
cat > "$OLD_DVAC/runtime/closed_for_pi05_grpo_pair_20260901.txt" <<EOF
closed_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
requested_action=stop old PPO-DVAC and replace with pi0.5 GRPO-DVAC
observed_state=already exited 255 before cutover after Ray node-memory monitor killed its workers
last_complete_step=$old_dvac_step
namespace=RLinf_1 already empty; no process signal was needed
EOF

setup_env
start_runtime "$CONTROL_RUN" "$CONTROL_PACKET" 4,5 'pi0.5 GRPO Control: chunk-level ratio/clip; DVAC off'
control_start_epoch=$(date +%s)
nohup setsid bash "$CONTROL_RUN/runtime/wrapper.sh" "$CONTROL_RUN/runtime" "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${control[@]}" > "$CONTROL_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
control_pid=$!; printf '%s\n' "$control_pid" > "$CONTROL_RUN/runtime/wrapper.pid"; printf '%s\n' "$control_pid" > "$CONTROL_RUN/runtime/owned.pgid"
start_observer "$CONTROL_RUN" "$control_pid" 4 5
wait_namespace_ready RLinf "$control_pid"
control_job=$(gpu_unique_job 4,5 pi05_control 6)

test "$(namespace_count RLinf_1)" = 0
start_runtime "$DVAC_RUN" "$DVAC_PACKET" 6,7 'pi0.5 GRPO-DVAC Action-Adv [0.5,1.5]; M5/L3/recent5'
dvac_start_epoch=$(date +%s)
nohup setsid bash "$DVAC_RUN/runtime/wrapper.sh" "$DVAC_RUN/runtime" "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${dvac[@]}" > "$DVAC_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
dvac_pid=$!; printf '%s\n' "$dvac_pid" > "$DVAC_RUN/runtime/wrapper.pid"; printf '%s\n' "$dvac_pid" > "$DVAC_RUN/runtime/owned.pgid"
start_observer "$DVAC_RUN" "$dvac_pid" 6 7
wait_namespace_ready RLinf_1 "$dvac_pid"
dvac_job=$(gpu_unique_job 6,7 pi05_dvac 6)

test "$control_job" != "$dvac_job"
kill -0 "$control_pid"; kill -0 "$dvac_pid"
test "$(namespace_count RLinf)" = 15; test "$(namespace_count RLinf_1)" = 15
printf 'old_control_step=%s old_dvac_step=%s\n' "$old_control_step" "$old_dvac_step"
printf 'control_run=%s\ncontrol_pid=%s\ncontrol_job=%s\ncontrol_gap_seconds=%s\n' "$CONTROL_RUN" "$control_pid" "$control_job" "$((control_start_epoch-cutover_epoch))"
printf 'dvac_run=%s\ndvac_pid=%s\ndvac_job=%s\ndvac_gap_seconds=%s\n' "$DVAC_RUN" "$dvac_pid" "$dvac_job" "$((dvac_start_epoch-cutover_epoch))"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_PPO_PAIR_TO_PI05_GRPO_DUAL_CUTOVER_OK
