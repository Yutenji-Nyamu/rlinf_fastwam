#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1

test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test ! -e "$RUN"

if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'GPU compute process already present on physical GPU 4-7; not starting formal PPO' >&2
  exit 1
fi
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  printf '%s\n' 'Existing Ray cluster present; not starting formal PPO' >&2
  exit 1
fi

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROOT/examples/embodiment"
export PYTHONPATH="$ROOT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "$RUN"
PPO_PLACEMENT='cluster.component_placement={actor\,\ env\,\ rollout:4-7}'
ARGS=(
  --config-path "$ROOT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_ppo_openpi
  "$PPO_PLACEMENT"
  "runner.logger.log_path=$RUN"
  "runner.max_epochs=1000"
  "runner.max_steps=100"
  "runner.val_check_interval=10"
  "runner.save_interval=10"
  "runner.resume_dir=null"
  "algorithm.update_epoch=2"
  "env.train.total_num_envs=128"
  "env.train.rollout_epoch=4"
  "env.train.max_episode_steps=200"
  "env.train.max_steps_per_rollout_epoch=200"
  "env.train.assets_path=$ROBOTWIN"
  "env.eval.total_num_envs=64"
  "env.eval.rollout_epoch=1"
  "env.eval.max_episode_steps=200"
  "env.eval.max_steps_per_rollout_epoch=200"
  "env.eval.use_fixed_reset_state_ids=true"
  "env.eval.assets_path=$ROBOTWIN"
  "actor.micro_batch_size=32"
  "actor.global_batch_size=2048"
  "actor.model.model_path=$MODEL"
)

"$VENV/bin/python" "$ROOT/examples/embodiment/train_embodied_agent.py" \
  "${ARGS[@]}" --cfg job --resolve > "$RUN/resolved.yaml"
sha256sum "$RUN/resolved.yaml" > "$RUN/resolved.yaml.sha256"

printf 'launch_time=%s\n' "$(date --iso-8601=seconds)" > "$RUN/launch_manifest.txt"
printf 'root_head=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" >> "$RUN/launch_manifest.txt"
printf 'robotwin_head=%s\n' "$(git -C "$ROBOTWIN" rev-parse HEAD)" >> "$RUN/launch_manifest.txt"
printf 'physical_gpus=4,5,6,7\n' >> "$RUN/launch_manifest.txt"
printf 'train_envs=128\neval_envs=64\nmax_steps=100\n' >> "$RUN/launch_manifest.txt"
printf 'train_rollout_epoch=4\ntrain_max_steps=200\n' >> "$RUN/launch_manifest.txt"
printf 'global_batch=2048\nmicro_batch=32\nupdate_epoch=2\n' >> "$RUN/launch_manifest.txt"
printf 'val_interval=10\nsave_interval=10\n' >> "$RUN/launch_manifest.txt"

nohup setsid "$VENV/bin/python" "$ROOT/examples/embodiment/train_embodied_agent.py" \
  "${ARGS[@]}" > "$RUN/driver.log" 2>&1 < /dev/null &
DRIVER_PID=$!
printf '%s\n' "$DRIVER_PID" > "$RUN/driver.pid"

sleep 3
kill -0 "$DRIVER_PID"
printf 'run=%s\n' "$RUN"
printf 'driver_pid=%s\n' "$DRIVER_PID"
printf 'resolved_sha256=%s\n' "$(awk '{print $1}' "$RUN/resolved.yaml.sha256")"
printf '%s\n' RLINF_PPO_FORMAL100_LAUNCHED
