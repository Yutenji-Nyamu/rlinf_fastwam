#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RUN_ID=pi0-adjust_bottle-fixed64-800baf80-v1
RUN=/data/chenyiteng/results/dvac-observation/$RUN_ID

test "$(git -C "$WT" rev-parse HEAD)" = 800baf80d6eab64169cf0e691eb04a681a093ee9
test -z "$(git -C "$WT" status --porcelain)"
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test ! -e "$RUN"

for gpu in 0 1 2 3; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
    printf 'physical GPU %s is not idle\n' "$gpu" >&2
    exit 1
  fi
done
if pgrep -u "$(id -u)" -x raylet >/dev/null || pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
  printf '%s\n' 'chenyiteng already has a live Ray cluster' >&2
  exit 1
fi

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "$RUN"
ARGS=(
  --config-path "$WT/evaluations/robotwin"
  --config-name robotwin_adjust_bottle_openpi_dvac_eval
  'cluster.component_placement={env\, rollout:0-3}'
  "runner.logger.log_path=$RUN"
  "rollout.model.model_path=$MODEL"
  "env.eval.assets_path=$ROBOTWIN"
  'env.eval.total_num_envs=64'
  'env.eval.rollout_epoch=1'
  'env.eval.max_episode_steps=200'
  'env.eval.max_steps_per_rollout_epoch=200'
  'env.eval.use_fixed_reset_state_ids=true'
  'rollout.dvac_telemetry.enabled=true'
  "rollout.dvac_telemetry.run_id=$RUN_ID"
  'rollout.dvac_telemetry.source_commit=800baf80d6eab64169cf0e691eb04a681a093ee9'
  'rollout.dvac_telemetry.seed_file_sha256=194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f'
  'rollout.dvac_telemetry.launch_command=verified_password_ssh command-file fixed64 v1'
)

"$VENV/bin/python" "$WT/evaluations/eval_embodied_agent.py" \
  "${ARGS[@]}" --cfg job --resolve > "$RUN/resolved.yaml"
sha256sum "$RUN/resolved.yaml" > "$RUN/resolved.yaml.sha256"

printf 'launch_time=%s\n' "$(date --iso-8601=seconds)" > "$RUN/launch_manifest.txt"
printf 'source_head=%s\nphysical_gpus=0,1,2,3\nepisodes=64\nmax_action_slots=12800\nmax_queries=256\n' \
  "$(git -C "$WT" rev-parse HEAD)" >> "$RUN/launch_manifest.txt"

nohup setsid timeout --signal=INT --kill-after=120s 3600s \
  "$VENV/bin/python" "$WT/evaluations/eval_embodied_agent.py" \
  "${ARGS[@]}" > "$RUN/driver.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN/driver.pid"

sleep 10
kill -0 "$pid"
printf 'run=%s\npid=%s\nresolved_sha256=%s\n' \
  "$RUN" "$pid" "$(awk '{print $1}' "$RUN/resolved.yaml.sha256")"
tail -n 20 "$RUN/driver.log" || true
printf '%s\n' 'SZ_PI0_DVAC_FIXED64_LAUNCHED'
