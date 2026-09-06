set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
HEAD=bab221afb8bedc32a8f01b171901a347f0258063
OLD_A="$ROOT/packets/b1-adjust-badseed-retry-m10-phys4-v12"
A_NAME=b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3
A_PACKET="$ROOT/packets/$A_NAME"
A_RUN="$ROOT/smokes/$A_NAME"

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$A_PACKET"
test ! -e "$A_RUN"
mkdir -p "$A_PACKET"
cp "$OLD_A/eval-seeds-adjust-bad1001.json" "$A_PACKET/eval-seeds-adjust-bad1001.json"
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status > "$A_PACKET/ray-status-before.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > "$A_PACKET/gpu-before.csv"
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo > "$A_PACKET/memory-before.txt"
df -h / /home /data > "$A_PACKET/disk-before.txt"
printf '%s\n' "$HEAD" > "$A_PACKET/source-head.txt"

cat > "$A_PACKET/command.txt" <<EOF
$VENV/bin/python $WT/evaluations/eval_embodied_agent.py --config-path $WT/examples/embodiment/config --config-name robotwin_move_stapler_pad_grpo_openpi_pi05_sidney '~cluster.component_placement' '+cluster.component_placement={env:4,rollout:4}' runner.task_type=embodied_eval runner.logger.log_path=$A_RUN runner.logger.experiment_name=pi05_sidney_b1_adjust_badseed_retry_m10_phys4_evalrunner_v13r3 runner.only_eval=true runner.val_check_interval=-1 runner.save_interval=-1 runner.resume_dir=null env.eval.total_num_envs=1 env.eval.rollout_epoch=1 env.eval.max_episode_steps=400 env.eval.max_steps_per_rollout_epoch=400 env.eval.use_fixed_reset_state_ids=true env.eval.seeds_path=$A_PACKET/eval-seeds-adjust-bad1001.json env.eval.assets_path=$ROBOTWIN env.eval.video_cfg.save_video=true env.eval.video_cfg.video_base_dir=$A_RUN/video/eval env.eval.task_config.save_path=$A_RUN/robotwin_data/eval env.eval.task_config.task_name=adjust_bottle env.eval.task_config.step_lim=400 actor.model.model_path=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab actor.model.num_steps=10 actor.model.openpi.num_steps=10
EOF

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
eval "$(cat "$A_PACKET/command.txt") --cfg job --resolve" > "$A_PACKET/resolved.yaml"

"$VENV/bin/python" - "$A_PACKET/resolved.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
assert c['runner']['task_type'] == 'embodied_eval'
assert c['runner']['only_eval'] is True
assert c['runner']['val_check_interval'] == -1
assert c['runner']['save_interval'] == -1
assert c['cluster']['component_placement'] == {'env': 4, 'rollout': 4}
assert 'actor' not in c['cluster']['component_placement']
assert c['rollout']['model'] == c['actor']['model']
m = c['rollout']['model']
assert m['model_type'] == 'openpi'
assert m['model_path'] == '/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab'
assert m['num_action_chunks'] == 50 and m['action_dim'] == 14 and m['num_steps'] == 10
assert m['openpi']['config_name'] == 'pi05_sidney_robotwin'
assert m['openpi']['action_chunk'] == 50 and m['openpi']['num_steps'] == 10
assert m['openpi']['num_images_in_input'] == 3 and m['openpi']['noise_level'] == 0.3
e = c['env']['eval']
assert e['total_num_envs'] == 1 and e['rollout_epoch'] == 1
assert e['max_episode_steps'] == 400 and e['max_steps_per_rollout_epoch'] == 400
assert e['task_config']['task_name'] == 'adjust_bottle' and e['task_config']['step_lim'] == 400
assert e['use_fixed_reset_state_ids'] is True
assert e['seeds_path'].endswith('/eval-seeds-adjust-bad1001.json')
print('A_V13_EVALRUNNER_RESOLVED_ASSERTIONS_OK')
PY

cat > "$A_PACKET/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/b1-adjust-badseed-retry-m10-phys4-evalrunner-v13r3
test "$(git -C "$WT" rev-parse HEAD)" = bab221afb8bedc32a8f01b171901a347f0258063
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then echo 'physical GPU4 is occupied; refusing Sidney B=1 eval smoke' >&2; exit 20; fi
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
chmod 0755 "$A_PACKET/run.sh"
sha256sum "$A_PACKET/run.sh" > "$A_PACKET/run-sha256.txt"
sha256sum "$A_PACKET/command.txt" > "$A_PACKET/command-sha256.txt"
printf '%s\n' prepared > "$A_PACKET/packet-complete.txt"
cat "$A_PACKET/run-sha256.txt" "$A_PACKET/command-sha256.txt"
echo SIDNEY_A_V13_PACKET_PREPARED
