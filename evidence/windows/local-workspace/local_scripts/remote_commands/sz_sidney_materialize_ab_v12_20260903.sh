set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
HEAD=bab221afb8bedc32a8f01b171901a347f0258063
OLD_A="$ROOT/packets/b1-adjust-badseed-retry-m10-phys4-v11"
OLD_B="$ROOT/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v11"
A_NAME=b1-adjust-badseed-retry-m10-phys4-v12
B_NAME=move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12
A_PACKET="$ROOT/packets/$A_NAME"
B_PACKET="$ROOT/packets/$B_NAME"
A_RUN="$ROOT/smokes/$A_NAME"
B_RUN="$ROOT/smokes/$B_NAME"

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
for path in "$A_PACKET" "$B_PACKET" "$A_RUN" "$B_RUN"; do test ! -e "$path"; done
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status > /tmp/sidney_ray_status_ab_v12_20260903.txt
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPUs 4/5 are occupied; refusing to freeze A/B v12' >&2
  exit 20
fi
mkdir -p "$A_PACKET" "$B_PACKET"
for packet in "$A_PACKET" "$B_PACKET"; do
  cp /tmp/sidney_ray_status_ab_v12_20260903.txt "$packet/ray-status-before.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > "$packet/gpu-before.csv"
  awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo > "$packet/memory-before.txt"
  df -h / /home /data > "$packet/disk-before.txt"
  printf '%s\n' "$HEAD" > "$packet/source-head.txt"
  printf '%s\n' prepared > "$packet/packet-complete.txt"
done

cp "$OLD_A/eval-seeds-adjust-bad1001.json" "$A_PACKET/eval-seeds-adjust-bad1001.json"
sed -e 's/b1-adjust-badseed-retry-m10-phys4-v11/b1-adjust-badseed-retry-m10-phys4-v12/g' \
    -e 's/pi05_sidney_b1_adjust_badseed_retry_m10_phys4_v11/pi05_sidney_b1_adjust_badseed_retry_m10_phys4_v12/g' \
    "$OLD_A/command.txt" > "$A_PACKET/command.txt"
sed -e 's/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v11/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12/g' \
    -e 's/pi05_sidney_move_grpo_smoke1_2gpu64x1_g8_b512_u2_m10_fixed8_phys45_localshard_v11/pi05_sidney_move_grpo_smoke1_2gpu64x1_g8_b512_u2_m10_fixed8_phys45_localshard_v12/g' \
    "$OLD_B/command.txt" > "$B_PACKET/command.txt"

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
eval "$(cat "$A_PACKET/command.txt") --cfg job --resolve" > "$A_PACKET/resolved.yaml"
eval "$(cat "$B_PACKET/command.txt") --cfg job --resolve" > "$B_PACKET/resolved.yaml"

"$VENV/bin/python" - "$A_PACKET/resolved.yaml" "$B_PACKET/resolved.yaml" <<'PY'
import sys, yaml
a=yaml.safe_load(open(sys.argv[1]))
b=yaml.safe_load(open(sys.argv[2]))
for c in (a,b):
 assert c['rollout']['model'] == c['actor']['model']
 m=c['actor']['model']
 assert m['model_type']=='openpi' and m['num_action_chunks']==50 and m['action_dim']==14 and m['num_steps']==10
 assert m['openpi']['config_name']=='pi05_sidney_robotwin' and m['openpi']['action_chunk']==50 and m['openpi']['num_steps']==10
 assert m['openpi']['num_images_in_input']==3 and m['openpi']['noise_level']==0.3
 assert c['env']['eval']['max_episode_steps']==400 and c['env']['eval']['task_config']['step_lim']==400
assert a['runner']['only_eval'] is True and a['env']['eval']['total_num_envs']==1
assert a['cluster']['component_placement']=={'actor, env, rollout':'4'}
assert b['runner']['only_eval'] is False and b['runner']['max_steps']==1
assert b['cluster']['component_placement']=={'actor, env, rollout':'4,5'}
assert b['env']['train']['total_num_envs']==64 and b['env']['train']['rollout_epoch']==1
assert b['algorithm']['group_size']==8 and b['algorithm']['update_epoch']==2
assert b['actor']['global_batch_size']==512 and b['actor']['micro_batch_size']==32
assert b['env']['eval']['total_num_envs']==8 and b['runner']['save_interval']==1
print('AB_V12_RESOLVED_ASSERTIONS_OK')
PY

cp "$OLD_A/contract.json" "$A_PACKET/contract.json"
cp "$OLD_B/contract.json" "$B_PACKET/contract.json"
cp "$OLD_A/model-sha256.txt" "$A_PACKET/model-sha256.txt"
cp "$OLD_B/model-sha256.txt" "$B_PACKET/model-sha256.txt"

cat > "$A_PACKET/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/b1-adjust-badseed-retry-m10-phys4-v12
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/b1-adjust-badseed-retry-m10-phys4-v12
test "$(git -C "$WT" rev-parse HEAD)" = bab221afb8bedc32a8f01b171901a347f0258063
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then echo 'physical GPU4 is occupied; refusing Sidney B=1 smoke' >&2; exit 20; fi
mkdir -p "$RUN/runtime"
cd "$WT"
source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
monitor() { while true; do date --iso-8601=seconds; nvidia-smi -i 4 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; awk '/MemAvailable:/ {print}' /proc/meminfo; sleep 30; done; }
monitor > "$RUN/runtime/resources.log" 2>&1 & MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
set +e; bash -e "$PACKET/command.txt" > "$RUN/runtime/driver.log" 2>&1; RC=$?; set -e
printf '%s\n' "$RC" > "$RUN/runtime/exit_code.txt"
exit "$RC"
EOF

cat > "$B_PACKET/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12
test "$(git -C "$WT" rev-parse HEAD)" = bab221afb8bedc32a8f01b171901a347f0258063
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then echo 'physical GPU4/5 is occupied; refusing Sidney GRPO smoke' >&2; exit 20; fi
mkdir -p "$RUN/runtime"
cd "$WT"
source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
monitor() { while true; do date --iso-8601=seconds; nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; awk '/MemAvailable:/ {print}' /proc/meminfo; sleep 30; done; }
monitor > "$RUN/runtime/resources.log" 2>&1 & MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
set +e; bash -e "$PACKET/command.txt" > "$RUN/runtime/driver.log" 2>&1; RC=$?; set -e
printf '%s\n' "$RC" > "$RUN/runtime/exit_code.txt"
exit "$RC"
EOF

chmod 0755 "$A_PACKET/run.sh" "$B_PACKET/run.sh"
sha256sum "$A_PACKET/run.sh" > "$A_PACKET/run-sha256.txt"
sha256sum "$B_PACKET/run.sh" > "$B_PACKET/run-sha256.txt"
sha256sum "$A_PACKET/command.txt" "$B_PACKET/command.txt" > "$ROOT/packets/ab-v12-command-sha256.txt"
cat "$A_PACKET/run-sha256.txt" "$B_PACKET/run-sha256.txt"
echo SIDNEY_AB_V12_PACKET_PREPARED
