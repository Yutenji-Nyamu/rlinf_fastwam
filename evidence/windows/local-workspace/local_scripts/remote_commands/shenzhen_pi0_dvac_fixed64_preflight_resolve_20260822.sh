#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RUN_ID=pi0-adjust_bottle-fixed64-800baf80-v1
RUN=/data/chenyiteng/results/dvac-observation/$RUN_ID
PACKET=/data/chenyiteng/results/dvac-observation/packets/$RUN_ID
LAUNCHER=$PACKET/shenzhen_pi0_dvac_launch_fixed64_gpu0_3_20260822.sh
EXPECTED_LAUNCHER_SHA=e7bdc9242b8dbd6c45f44eefa887390988557b4778cc2e2b57209b4b99a8399b

test -d "$PACKET"
test -f "$LAUNCHER"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
bash -n "$LAUNCHER"
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
test "$(df -PB1 /data | awk 'NR==2 {print $4}')" -ge 2147483648

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=''
export NVIDIA_VISIBLE_DEVICES=none
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

RESOLVED_TMP="$PACKET/resolved.yaml.tmp"
RESOLVED="$PACKET/resolved.yaml"
test ! -e "$RESOLVED"
"$VENV/bin/python" "$WT/evaluations/eval_embodied_agent.py" \
  "${ARGS[@]}" --cfg job --resolve > "$RESOLVED_TMP"
mv "$RESOLVED_TMP" "$RESOLVED"
sha256sum "$RESOLVED" > "$PACKET/resolved.yaml.sha256"

"$VENV/bin/python" - "$RESOLVED" <<'PY'
import pathlib
import sys
import yaml

p = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
assert cfg["cluster"]["component_placement"] == {"env": "0-3", "rollout": "0-3"}
assert cfg["env"]["eval"]["total_num_envs"] == 64
assert cfg["env"]["eval"]["rollout_epoch"] == 1
assert cfg["env"]["eval"]["max_episode_steps"] == 200
assert cfg["env"]["eval"]["max_steps_per_rollout_epoch"] == 200
assert cfg["env"]["eval"]["use_fixed_reset_state_ids"] is True
assert cfg["rollout"]["dvac_telemetry"]["enabled"] is True
assert cfg["rollout"]["dvac_telemetry"]["run_id"] == "pi0-adjust_bottle-fixed64-800baf80-v1"
assert cfg["rollout"]["dvac_telemetry"]["source_commit"] == "800baf80d6eab64169cf0e691eb04a681a093ee9"
assert cfg["rollout"]["model"]["model_path"].endswith("@92684e50")
assert cfg["runner"]["logger"]["log_path"].endswith("pi0-adjust_bottle-fixed64-800baf80-v1")
PY

printf 'launcher_sha256=%s\n' "$EXPECTED_LAUNCHER_SHA"
printf 'resolved_sha256=%s\n' "$(awk '{print $1}' "$PACKET/resolved.yaml.sha256")"
printf 'data_available_bytes=%s\n' "$(df -PB1 /data | awk 'NR==2 {print $4}')"
printf 'host_mem_available_kib=%s\n' "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' 'SZ_PI0_DVAC_FIXED64_PREFLIGHT_RESOLVED_OK'
