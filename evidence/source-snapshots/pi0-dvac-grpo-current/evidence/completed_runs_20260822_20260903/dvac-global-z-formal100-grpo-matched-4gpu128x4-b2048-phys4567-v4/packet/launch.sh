#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=66c863bc5a45e90cb5161b30af54355b1104c810
BRANCH=codex/sz-current-pi0-dvac-grpo
RAY_ADDRESS=172.17.0.1:6389
NAME=dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/$NAME
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/$NAME
EXPERIMENT=robotwin_grpo_openpi_dvac_global_z_matched
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" rev-parse "personal/$BRANCH")" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test ! -e "$RUN"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
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
ARGS=(
  --config-path "$WT/examples/embodiment/config" --config-name robotwin_adjust_bottle_grpo_openpi
  'cluster.component_placement={actor\, env\, rollout:"4,5,6,7"}'
  "runner.logger.log_path=$RUN" "runner.logger.experiment_name=$EXPERIMENT"
  runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=10 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=2 algorithm.dvac_gradient_weighting.mode=apply
  env.train.total_num_envs=128 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true "env.train.video_cfg.video_base_dir=$RUN/video/train" "env.train.task_config.save_path=$RUN/robotwin_data/train"
  env.eval.total_num_envs=64 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200 env.eval.use_fixed_reset_state_ids=true
  "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true "env.eval.video_cfg.video_base_dir=$RUN/video/eval" "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
  actor.micro_batch_size=32 actor.global_batch_size=2048 "actor.model.model_path=$MODEL"
)
mkdir -p "$RUN/runtime"
cp "$PACKET/resolved.yaml" "$PACKET/contract.json" "$PACKET/command.txt" "$PACKET/comparison_to_shenzhen_grpo_v2.json" "$RUN/runtime/"
printf '%s\n' "started_at=$(date --iso-8601=seconds)" "source_head=$HEAD" 'physical_gpus=4,5,6,7' \
  'train=128 env x 4 epochs = 512 trajectories/step; G8; max 2048 chunk records/step' \
  'actor=GB2048/MB32/update2; max 2 optimizer calls/step' 'dvac=L3; recent5; warmup1; weights[0,2]' \
  'eval=fixed64 every10; checkpoint every10' 'comparison=all non-DVAC mathematical fields match Shenzhen GRPO v2 resolved' \
  'normal_stop=step100' 'hard_timeout=129600s' > "$RUN/runtime/launch_manifest.txt"
cat > "$RUN/runtime/wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set +e
runtime=$1; shift
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 129600s "$@" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
WRAP
chmod 700 "$RUN/runtime/wrapper.sh"
nohup setsid bash "$RUN/runtime/wrapper.sh" "$RUN/runtime" "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" > "$RUN/runtime/wrapper.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/runtime/wrapper.pid"
printf '%s\n' "$pid" > "$RUN/runtime/owned.pgid"
cat > "$RUN/runtime/observer.sh" <<'OBS'
#!/usr/bin/env bash
set -u
pid=$1; out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct,gpu6_used_mib,gpu6_util_pct,gpu7_used_mib,gpu7_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
OBS
chmod 700 "$RUN/runtime/observer.sh"
nohup setsid bash "$RUN/runtime/observer.sh" "$pid" "$RUN/runtime/resource.csv" > "$RUN/runtime/observer.log" 2>&1 < /dev/null &
observer=$!
printf '%s\n' "$observer" > "$RUN/runtime/observer.pid"
sleep 15
kill -0 "$pid"
kill -0 "$observer"
printf 'run=%s\nwrapper_pid=%s\nobserver_pid=%s\n' "$RUN" "$pid" "$observer"
tail -n 30 "$RUN/runtime/driver.log" || true
echo 'SZ_DVAC_GRPO_MATCHED128X4_V4_LAUNCHED'
