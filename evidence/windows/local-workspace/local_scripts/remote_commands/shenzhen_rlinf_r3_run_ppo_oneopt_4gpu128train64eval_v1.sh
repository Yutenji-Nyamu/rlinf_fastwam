#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1
CKPT="$RUN/robotwin_ppo_openpi/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test -s "$MODEL/model-00001-of-00002.safetensors"
test ! -e "$RUN"
if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'GPU compute process already present on physical GPU 4-7; not starting PPO' >&2
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
  "runner.max_epochs=1"
  "runner.max_steps=1"
  "runner.val_check_interval=1"
  "runner.save_interval=1"
  "algorithm.update_epoch=1"
  "env.train.total_num_envs=128"
  "env.train.rollout_epoch=1"
  "env.train.max_steps_per_rollout_epoch=50"
  "env.train.assets_path=$ROBOTWIN"
  "env.eval.total_num_envs=64"
  "env.eval.rollout_epoch=1"
  "env.eval.max_episode_steps=200"
  "env.eval.max_steps_per_rollout_epoch=200"
  "env.eval.use_fixed_reset_state_ids=true"
  "env.eval.assets_path=$ROBOTWIN"
  "actor.micro_batch_size=32"
  "actor.global_batch_size=128"
  "actor.model.model_path=$MODEL"
)

"$VENV/bin/python" "$ROOT/examples/embodiment/train_embodied_agent.py" \
  "${ARGS[@]}" --cfg job --resolve > "$RUN/resolved.yaml"

/usr/bin/time -v timeout --signal=INT --kill-after=120s 7200s \
  "$VENV/bin/python" "$ROOT/examples/embodiment/train_embodied_agent.py" "${ARGS[@]}" \
  2>&1 | tee "$RUN/driver.log"

test -s "$CKPT"
test -s "$RUN/robotwin_ppo_openpi/checkpoints/global_step_1/actor/dcp_checkpoint/.metadata"
find "$RUN/robotwin_ppo_openpi/checkpoints/global_step_1/actor/dcp_checkpoint" \
  -type f -name '*.distcp' -size +0c -print -quit | grep -q .
test "$(find "$RUN/robotwin_ppo_openpi/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n')" = global_step_1
video=$(find "$RUN" -type f -name '*.mp4' -print -quit)
test -n "$video" && test -s "$video"
sha256sum "$RUN/resolved.yaml" "$RUN/driver.log" "$CKPT" "$video"
du -sh "$RUN/robotwin_ppo_openpi/checkpoints/global_step_1"
printf '%s\n' 'R3_PPO_ONEOPT_4GPU128TRAIN64EVAL_OK'
