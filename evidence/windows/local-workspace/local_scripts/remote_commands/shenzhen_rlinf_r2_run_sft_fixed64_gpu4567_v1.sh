#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1

test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test -s "$MODEL/model-00001-of-00002.safetensors"
test ! -e "$RUN"
if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'GPU compute process already present on physical GPU 4-7; not starting fixed-64' >&2
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
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "$RUN"
EVAL_PLACEMENT='cluster.component_placement={env\,\ rollout:4-7}'
ARGS=(
  --config-path "$ROOT/evaluations/robotwin"
  --config-name robotwin_adjust_bottle_openpi_eval
  "$EVAL_PLACEMENT"
  "runner.logger.log_path=$RUN"
  "rollout.model.model_path=$MODEL"
  "env.eval.assets_path=$ROBOTWIN"
  "env.eval.total_num_envs=64"
  "env.eval.rollout_epoch=1"
  "env.eval.max_episode_steps=200"
  "env.eval.max_steps_per_rollout_epoch=200"
  "env.eval.use_fixed_reset_state_ids=true"
)

"$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" \
  "${ARGS[@]}" --cfg job --resolve > "$RUN/resolved.yaml"

/usr/bin/time -v timeout --signal=INT --kill-after=120s 3600s \
  "$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" "${ARGS[@]}" \
  2>&1 | tee "$RUN/driver.log"

video=$(find "$RUN" -type f -name '*.mp4' -print -quit)
test -n "$video" && test -s "$video"
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames \
  -of default=noprint_wrappers=1 "$video"
sha256sum "$RUN/resolved.yaml" "$RUN/driver.log" "$video"
printf '%s\n' 'R2_SFT_FIXED64_GPU4567_OK'
