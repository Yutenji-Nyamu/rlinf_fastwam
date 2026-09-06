#!/usr/bin/env bash
set -euo pipefail

OLD=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
CONTROL=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=306ce2e98a06b6f439a1070d8942e20132e48d49
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
NAME=prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2
RUN="$ROOT/runs/$NAME"
PACKET="$ROOT/packets/$NAME"
OLD_NAMESPACE=RLinf_1
OLD_JOB=50000000
CONTROL_JOB=3e000000

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test ! -e "$RUN"
for file in resolved.yaml control_same_code_resolved.yaml parity.json contract.json command.txt; do test -s "$PACKET/$file"; done
grep -Fq 'checkpoint_format: local_shard' "$PACKET/resolved.yaml"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

control_pid=$(<"$CONTROL/runtime/wrapper.pid")
kill -0 "$control_pid"
test "$(ps -o user= -p "$control_pid" | xargs)" = chenyiteng

old_pid=$(<"$OLD/runtime/wrapper.pid")
old_pgid=$(<"$OLD/runtime/owned.pgid")
test "$old_pid" = "$old_pgid"
kill -0 "$old_pid"
test "$(ps -o user= -p "$old_pid" | xargs)" = chenyiteng
ps -o args= -p "$old_pid" | grep -F "$OLD/runtime/wrapper.sh" >/dev/null

old_gpu_count=0
while read -r process_pid; do
  [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
  job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
  [[ "$job" == "$OLD_JOB" ]] || { echo "foreign job on GPU6/7: pid=$process_pid job=$job" >&2; exit 21; }
  test "$(ps -o user= -p "$process_pid" | xargs)" = chenyiteng
  old_gpu_count=$((old_gpu_count + 1))
done < <(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
test "$old_gpu_count" -ge 2

while read -r process_pid; do
  [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
  job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
  [[ "$job" == "$CONTROL_JOB" ]] || { echo "unexpected job on GPU4/5: pid=$process_pid job=$job" >&2; exit 22; }
done < <(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)

last_complete_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$OLD/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
cutover_start_epoch=$(date +%s)
kill -TERM -- "-$old_pgid"
for _ in $(seq 1 60); do ! kill -0 "$old_pid" 2>/dev/null && break; sleep 1; done
if kill -0 "$old_pid" 2>/dev/null; then kill -KILL -- "-$old_pgid"; sleep 2; fi
! kill -0 "$old_pid" 2>/dev/null

observer=$(cat "$OLD/runtime/observer.pid" 2>/dev/null || true)
if [[ -n "$observer" ]] && kill -0 "$observer" 2>/dev/null; then
  kill -TERM -- "-$observer" 2>/dev/null || kill -TERM "$observer" 2>/dev/null || true
fi

RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$OLD_NAMESPACE" "$VENV/bin/python" - <<'PY'
import os, time, ray
target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_prism_checkpoint_fix", logging_level="ERROR")
def names():
    return sorted(row["name"] for row in ray.util.list_named_actors(all_namespaces=True) if row.get("namespace") == target)
targets = names()
print(f"target_namespace={target} actors_before={len(targets)}")
assert targets, "expected stuck Prism actors"
managers = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
for name in sorted(targets, key=lambda item: (item in managers, item)):
    try: ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
    except ValueError: pass
for _ in range(120):
    if not names(): break
    time.sleep(1)
else: raise SystemExit(f"namespace cleanup incomplete: {names()}")
ray.shutdown()
PY

for _ in $(seq 1 120); do
  target_gpu=0
  while read -r process_pid; do
    [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ "$job" == "$OLD_JOB" ]] && target_gpu=$((target_gpu + 1))
  done < <(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  (( target_gpu == 0 )) && break
  sleep 1
done
test "$target_gpu" -eq 0
kill -0 "$control_pid"

cat > "$OLD/runtime/stopped_by_user_for_checkpoint_fix.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=Step10 DCP optimizer-state collective stalled; user authorized fresh local_shard restart
last_complete_step=$last_complete_step
cleanup=exact owned PGID and namespace $OLD_NAMESPACE; Control/shared Ray/other users unchanged
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
  'cluster.component_placement={actor\, env\, rollout:"6,7"}'
  "runner.logger.log_path=$RUN"
  runner.logger.experiment_name=robotwin_prism_dvac_rank_rloo_formal100_2gpu64x4_b1024_fixed32_eval5_localshard_phys67_v2
  runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=2 algorithm.adv_type=prism_rloo algorithm.filter_rewards=false
  algorithm.dvac_gradient_weighting.mode=off algorithm.dvac_gradient_weighting.weight_min=null algorithm.dvac_gradient_weighting.weight_max=null
  algorithm.prism_dvac.enabled=true algorithm.prism_dvac.selected_l=3 algorithm.prism_dvac.quality_lambda=0.2 algorithm.prism_dvac.log_eps=1.0e-12
  env.train.total_num_envs=64 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true "env.train.video_cfg.video_base_dir=$RUN/video/train" "env.train.task_config.save_path=$RUN/robotwin_data/train"
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200 env.eval.use_fixed_reset_state_ids=true
  "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true "env.eval.video_cfg.video_base_dir=$RUN/video/eval" "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
  actor.micro_batch_size=32 actor.global_batch_size=1024 +actor.fsdp_config.checkpoint_format=local_shard "actor.model.model_path=$MODEL"
)

mkdir -p "$RUN/runtime"
cp "$PACKET/resolved.yaml" "$PACKET/control_same_code_resolved.yaml" "$PACKET/parity.json" "$PACKET/contract.json" "$PACKET/command.txt" "$RUN/runtime/"
new_started_epoch=$(date +%s)
cat > "$RUN/runtime/launch_manifest.txt" <<EOF
started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
source_head=$HEAD
physical_gpus=6,7
fresh_start=true; target_step=100
train=64 env x 4 epochs = 256 trajectories; G8; max 1024 chunk records
actor=GB1024/MB32/update2
method=Prism DVAC rank RLOO; ST off
eval=fixed32 every5; checkpoint=local_shard every10
cutover_gap_seconds=$((new_started_epoch-cutover_start_epoch))
EOF

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
nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
new_pid=$!
printf '%s\n' "$new_pid" > "$RUN/runtime/wrapper.pid"
printf '%s\n' "$new_pid" > "$RUN/runtime/owned.pgid"

cat > "$RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,gpu6_used_mib,gpu6_util_pct,gpu7_used_mib,gpu7_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
chmod 700 "$RUN/runtime/observer.sh"
nohup setsid bash "$RUN/runtime/observer.sh" "$new_pid" "$RUN/runtime/resource.csv" > "$RUN/runtime/observer.log" 2>&1 < /dev/null &
observer=$!
printf '%s\n' "$observer" > "$RUN/runtime/observer.pid"

sleep 15
kill -0 "$new_pid"
kill -0 "$control_pid"
printf 'old_last_complete_step=%s\nnew_run=%s\nnew_wrapper_pid=%s\nobserver_pid=%s\ncutover_gap_seconds=%s\n' "$last_complete_step" "$RUN" "$new_pid" "$observer" "$((new_started_epoch-cutover_start_epoch))"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_PRISM_LOCALSHARD_FORMAL100_V2_RESTARTED

