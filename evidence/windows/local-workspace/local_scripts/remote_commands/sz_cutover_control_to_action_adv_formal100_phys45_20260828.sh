#!/usr/bin/env bash
set -euo pipefail

CONTROL=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
PRISM=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=a5b94b6f10a9212502d6930f07543f61e31af52e
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
NAME=dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
RUN="$ROOT/runs/$NAME"
PACKET="$ROOT/packets/$NAME"
CONTROL_NAMESPACE=RLinf
CONTROL_JOB=3e000000
PRISM_NAMESPACE=RLinf_1
PRISM_JOB=5a000000

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test ! -e "$RUN"
for file in resolved.yaml control_same_code_resolved.yaml parity.json contract.json command.txt packet_complete.txt; do
  test -s "$PACKET/$file"
done
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

control_pid=$(<"$CONTROL/runtime/wrapper.pid")
control_pgid=$(<"$CONTROL/runtime/owned.pgid")
test "$control_pid" = "$control_pgid"
kill -0 "$control_pid"
test "$(ps -o user= -p "$control_pid" | xargs)" = chenyiteng
ps -o args= -p "$control_pid" | grep -F "$CONTROL/runtime/wrapper.sh" >/dev/null

prism_pid=$(<"$PRISM/runtime/wrapper.pid")
prism_pgid=$(<"$PRISM/runtime/owned.pgid")
test "$prism_pid" = "$prism_pgid"
kill -0 "$prism_pid"
test "$(ps -o user= -p "$prism_pid" | xargs)" = chenyiteng
ps -o args= -p "$prism_pid" | grep -F "$PRISM/runtime/wrapper.sh" >/dev/null

check_gpu_job() {
  local devices=$1 expected=$2 minimum=$3 label=$4 count=0
  while read -r process_pid; do
    [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ "$job" == "$expected" ]] || { echo "foreign job on GPU$devices: pid=$process_pid job=$job expected=$expected" >&2; exit 21; }
    test "$(ps -o user= -p "$process_pid" | xargs)" = chenyiteng
    count=$((count + 1))
  done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  test "$count" -ge "$minimum"
  printf '%s_gpu_processes=%s job=%s\n' "$label" "$count" "$expected"
}

check_gpu_job 4,5 "$CONTROL_JOB" 6 control
check_gpu_job 6,7 "$PRISM_JOB" 6 prism

RAY_ADDRESS="$RAY_ADDRESS" CONTROL_NAMESPACE="$CONTROL_NAMESPACE" PRISM_NAMESPACE="$PRISM_NAMESPACE" "$VENV/bin/python" - <<'PY'
import os
import ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_action_adv_cutover_precheck", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
def names(ns):
    return sorted(row["name"] for row in rows if row.get("namespace") == ns)
control = names(os.environ["CONTROL_NAMESPACE"])
prism = names(os.environ["PRISM_NAMESPACE"])
assert len(control) == 15, control
assert len(prism) == 15, prism
print(f"control_namespace_actors={len(control)} prism_namespace_actors={len(prism)}")
ray.shutdown()
PY

cutover_start_epoch=$(date +%s)
last_complete_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$CONTROL/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)

kill -TERM -- "-$control_pgid"
for _ in $(seq 1 60); do
  if ! kill -0 "$control_pid" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$control_pid" 2>/dev/null; then
  kill -KILL -- "-$control_pgid"
  sleep 2
fi
! kill -0 "$control_pid" 2>/dev/null

observer=$(cat "$CONTROL/runtime/observer.pid" 2>/dev/null || true)
if [[ -n "$observer" ]] && kill -0 "$observer" 2>/dev/null; then
  kill -TERM -- "-$observer" 2>/dev/null || kill -TERM "$observer" 2>/dev/null || true
fi

RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$CONTROL_NAMESPACE" PRESERVE_NAMESPACE="$PRISM_NAMESPACE" "$VENV/bin/python" - <<'PY'
import os
import time
import ray
target = os.environ["TARGET_NAMESPACE"]
preserve = os.environ["PRESERVE_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_control_to_action_adv", logging_level="ERROR")
def names(ns):
    return sorted(row["name"] for row in ray.util.list_named_actors(all_namespaces=True) if row.get("namespace") == ns)
targets = names(target)
preserved_before = names(preserve)
assert len(targets) == 15, targets
assert len(preserved_before) == 15, preserved_before
manager_names = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
for name in sorted(targets, key=lambda item: (item in manager_names, item)):
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
    raise SystemExit(f"target namespace cleanup incomplete: {remaining}")
assert names(preserve) == preserved_before, "Prism named actor set changed during Control cleanup"
print(f"target_namespace={target} actors_after=0 preserved_namespace={preserve} actors={len(preserved_before)}")
ray.shutdown()
PY

for _ in $(seq 1 120); do
  target_gpu=0
  while read -r process_pid; do
    [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ "$job" == "$CONTROL_JOB" ]] && target_gpu=$((target_gpu + 1))
  done < <(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  (( target_gpu == 0 )) && break
  sleep 1
done
test "$target_gpu" -eq 0
kill -0 "$prism_pid"
check_gpu_job 6,7 "$PRISM_JOB" 6 prism_preserved

cat > "$CONTROL/runtime/stopped_by_user_for_action_adv.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested replacement of the two-GPU clean GRPO Control by GRPO-DVAC Action-Adv [0,2]
last_complete_step=$last_complete_step
ray_job_id=$CONTROL_JOB
namespace=$CONTROL_NAMESPACE
cleanup=exact owned PGID and exact named actors; Prism/shared Ray/other users unchanged
EOF

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

args=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_grpo_openpi
  'cluster.component_placement={actor\, env\, rollout:"4,5"}'
  "runner.logger.log_path=$RUN"
  runner.logger.experiment_name=robotwin_dvac_action_adv_w0to2_formal100_2gpu64x4_b1024_fixed32_eval5_phys45_v1
  runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=2 algorithm.adv_type=grpo algorithm.filter_rewards=true
  algorithm.logprob_type=action_level
  algorithm.dvac_gradient_weighting.mode=apply
  algorithm.dvac_gradient_weighting.application=action_advantage
  algorithm.dvac_gradient_weighting.selected_l=3
  algorithm.dvac_gradient_weighting.warmup_steps=1
  algorithm.dvac_gradient_weighting.window_steps=5
  algorithm.dvac_gradient_weighting.weight_min=0.0
  algorithm.dvac_gradient_weighting.weight_max=2.0
  env.train.total_num_envs=64 env.train.rollout_epoch=4
  env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN"
  env.train.video_cfg.save_video=true
  "env.train.video_cfg.video_base_dir=$RUN/video/train"
  "env.train.task_config.save_path=$RUN/robotwin_data/train"
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true
  "env.eval.assets_path=$ROBOTWIN"
  env.eval.video_cfg.save_video=true
  "env.eval.video_cfg.video_base_dir=$RUN/video/eval"
  "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
  actor.micro_batch_size=32 actor.global_batch_size=1024
  "actor.model.model_path=$MODEL"
)

mkdir -p "$RUN/runtime"
cp "$PACKET"/{resolved.yaml,control_same_code_resolved.yaml,parity.json,contract.json,command.txt,packet_complete.txt} "$RUN/runtime/"
new_started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
new_started_epoch=$(date +%s)
printf '%s\n' \
  "started_at=$new_started_at" \
  "source_head=$HEAD" \
  'physical_gpus=4,5' \
  'fresh_start=true; target_step=100' \
  'train=64 env x 4 epochs = 256 trajectories/step; G8; max1024 records' \
  'actor=GB1024/MB32/update2' \
  'method=trajectory GRPO A times raw [0,2] DVAC per-action weight; action-level ratio/clip' \
  'eval=fixed32 every5; checkpoint=DCP every10' \
  'normal_stop=complete step100/checkpoint/exit0; hard_timeout=216000s' \
  "cutover_gap_seconds=$((new_started_epoch-cutover_start_epoch))" \
  > "$RUN/runtime/launch_manifest.txt"

cat > "$RUN/runtime/wrapper.sh" <<'WRAP'
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
chmod 700 "$RUN/runtime/wrapper.sh"
nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" \
  > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
new_pid=$!
printf '%s\n' "$new_pid" > "$RUN/runtime/wrapper.pid"
printf '%s\n' "$new_pid" > "$RUN/runtime/owned.pgid"

cat > "$RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 4,5 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
chmod 700 "$RUN/runtime/observer.sh"
nohup setsid bash "$RUN/runtime/observer.sh" "$new_pid" "$RUN/runtime/resource.csv" > "$RUN/runtime/observer.log" 2>&1 < /dev/null &
new_observer=$!
printf '%s\n' "$new_observer" > "$RUN/runtime/observer.pid"

sleep 15
kill -0 "$new_pid"
kill -0 "$prism_pid"
printf 'control_last_complete_step=%s\nnew_run=%s\nnew_wrapper_pid=%s\nobserver_pid=%s\ncutover_gap_seconds=%s\n' \
  "$last_complete_step" "$RUN" "$new_pid" "$new_observer" "$((new_started_epoch-cutover_start_epoch))"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_CONTROL_TO_ACTION_ADV_FORMAL100_CUTOVER_OK
