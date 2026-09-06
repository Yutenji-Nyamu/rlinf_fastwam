#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
RAY_ADDRESS=172.17.0.1:6389
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
CONTROL=grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
DVAC=dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
for name in "$CONTROL" "$DVAC"; do
  test ! -e "$ROOT/runs/$name"
  test -s "$ROOT/packets/$name/resolved.yaml"
  test -s "$ROOT/packets/$name/contract.json"
  test -s "$ROOT/packets/$name/command.txt"
  test -s "$ROOT/packets/$name/pair_parity.json"
done
if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPUs 4,5,6,7 are not idle' >&2
  exit 20
fi

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

start_one() {
  local name=$1 placement=$2 experiment=$3 mode=$4
  local run="$ROOT/runs/$name" packet="$ROOT/packets/$name"
  local -a args=(
    --config-path "$WT/examples/embodiment/config"
    --config-name robotwin_adjust_bottle_grpo_openpi
    "cluster.component_placement={actor\\, env\\, rollout:\"$placement\"}"
    "runner.logger.log_path=$run"
    "runner.logger.experiment_name=$experiment"
    runner.max_epochs=1000
    runner.max_steps=100
    runner.val_check_interval=5
    runner.save_interval=10
    runner.resume_dir=null
    algorithm.update_epoch=2
    "algorithm.dvac_gradient_weighting.mode=$mode"
    algorithm.dvac_gradient_weighting.weight_min=null
    algorithm.dvac_gradient_weighting.weight_max=null
    env.train.total_num_envs=64
    env.train.rollout_epoch=4
    env.train.max_episode_steps=200
    env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN"
    env.train.video_cfg.save_video=true
    "env.train.video_cfg.video_base_dir=$run/video/train"
    "env.train.task_config.save_path=$run/robotwin_data/train"
    env.eval.total_num_envs=32
    env.eval.rollout_epoch=1
    env.eval.max_episode_steps=200
    env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true
    "env.eval.assets_path=$ROBOTWIN"
    env.eval.video_cfg.save_video=true
    "env.eval.video_cfg.video_base_dir=$run/video/eval"
    "env.eval.task_config.save_path=$run/robotwin_data/eval"
    actor.micro_batch_size=32
    actor.global_batch_size=1024
    "actor.model.model_path=$MODEL"
  )

  mkdir -p "$run/runtime"
  cp "$packet/resolved.yaml" "$packet/contract.json" "$packet/command.txt" "$packet/pair_parity.json" "$run/runtime/"
  printf '%s\n' \
    "started_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" \
    "source_head=$HEAD" \
    "physical_gpus=$placement" \
    "method=$mode" \
    'fresh_start=true; target_step=100' \
    'train=64 env x 4 epochs = 256 trajectories/step; G8; max 1024 chunk records/step' \
    'actor=GB1024/MB32/update2; 512 records/rank; 2 optimizer calls/step' \
    'eval=fixed32 as 32 env x 1 wave every5; checkpoint every10' \
    'normal_stop=complete step100; hard_timeout=216000s' > "$run/runtime/launch_manifest.txt"

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
  nohup setsid bash "$run/runtime/wrapper.sh" "$run/runtime" \
    "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" \
    > "$run/runtime/wrapper.log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$run/runtime/wrapper.pid"
  printf '%s\n' "$pid" > "$run/runtime/owned.pgid"

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
  local gpu_a=${placement%,*} gpu_b=${placement#*,}
  nohup setsid bash "$run/runtime/observer.sh" "$pid" "$run/runtime/resource.csv" "$gpu_a" "$gpu_b" > "$run/runtime/observer.log" 2>&1 < /dev/null &
  local observer=$!
  printf '%s\n' "$observer" > "$run/runtime/observer.pid"
  printf 'name=%s wrapper_pid=%s observer_pid=%s\n' "$name" "$pid" "$observer"
}

start_one "$CONTROL" '4,5' robotwin_grpo_control_2gpu64x4_b1024_fixed32_eval5_v2 off

for _ in $(seq 1 120); do
  count=$(RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_wait_control_manager", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
print(sum(1 for row in rows if row.get("namespace") == "RLinf" and row.get("name") in {"NodeManager", "WorkerManager"}))
ray.shutdown()
PY
)
  [[ "$count" == 2 ]] && break
  sleep 1
done
test "$count" = 2

start_one "$DVAC" '6,7' robotwin_grpo_dvac_global_z_w0to2_2gpu64x4_b1024_fixed32_eval5_v2 apply

sleep 15
for name in "$CONTROL" "$DVAC"; do
  pid=$(<"$ROOT/runs/$name/runtime/wrapper.pid")
  kill -0 "$pid"
done
printf 'control_run=%s\n' "$ROOT/runs/$CONTROL"
printf 'dvac_run=%s\n' "$ROOT/runs/$DVAC"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo SZ_DUAL_GRPO_2GPU_FIXED32_FORMAL100_V2_LAUNCHED



