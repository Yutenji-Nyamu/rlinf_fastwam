set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
ROBOTWIN=/root/autodl-tmp/RoboTwin_RLinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
CFG=robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
CKPT1="$RUN_ROOT/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke/checkpoints/global_step_1"
CKPT2="$RUN_ROOT/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke/checkpoints/global_step_2"

test -s "$CKPT1/actor/local_shard_checkpoint/checkpoint_rank_0.pt"
test -s "$CKPT1/actor/local_shard_checkpoint/checkpoint_rank_1.pt"
test -s "$CKPT1/actor/sac_components/alpha/dcp_checkpoint/.metadata"
test -n "$(find "$CKPT1/actor/sac_components/alpha/dcp_checkpoint" -maxdepth 1 -type f -name '*.distcp' -print -quit)"
test -s "$CKPT1/actor/sac_components/target_model/checkpoint_rank_0.pt"
test -s "$CKPT1/actor/sac_components/target_model/checkpoint_rank_1.pt"
test -s "$CKPT1/actor/sac_components/dsrl_trainer_state_rank_0.pt"
test -s "$CKPT1/actor/sac_components/dsrl_trainer_state_rank_1.pt"
test -s "$CKPT1/actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt"
test -s "$CKPT1/actor/sac_components/replay_buffer/rank_1/dsrl_transition_replay.pt"
test ! -e "$CKPT2"
test ! -e "$RUN_ROOT/resume_driver.log"
test ! -e "$RUN_ROOT/resume.pid"
test ! -e "$RUN_ROOT/resource_monitor/resume"

LIVE_TARGETS=$(
  ps -eo pid=,comm=,args= |
    awk '$2 ~ /^(python|python3|raylet|gcs_server)$/ && $0 ~ /(train_embodied_agent|ray::|raylet|gcs_server|monitor_resources.py)/ {print}'
)
if test -n "$LIVE_TARGETS"; then
  echo "LIVE_TARGETS_FOUND"
  echo "$LIVE_TARGETS"
  exit 51
fi

if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  awk '$1 != 0 {bad=1} END {exit bad ? 0 : 1}'; then
  echo "GPU_MEMORY_NOT_ZERO"
  nvidia-smi
  exit 52
fi

echo "HASHING_CKPT1_START=$(date --iso-8601=seconds)"
find "$CKPT1" -type f -print0 |
  sort -z |
  xargs -0 sha256sum > "$RUN_ROOT/ckpt1_before_resume.sha256"
echo "HASHING_CKPT1_DONE=$(date --iso-8601=seconds)"
test -s "$RUN_ROOT/ckpt1_before_resume.sha256"

mkdir -p "$RUN_ROOT/resource_monitor/resume"
cat /sys/fs/cgroup/memory.events > "$RUN_ROOT/resource_monitor/resume/memory.events.baseline.txt"
awk '$1 ~ /^(anon|file|shmem|file_mapped|inactive_file|active_file|kernel_stack|pagetables|slab)$/ {print}' \
  /sys/fs/cgroup/memory.stat > "$RUN_ROOT/resource_monitor/resume/memory.stat.baseline.txt"
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits \
  > "$RUN_ROOT/resource_monitor/resume/gpu.baseline.csv"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="$REPO/examples/embodiment"
export REPO_PATH="$REPO"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$REPO:$ROBOTWIN"
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

RESUME_CMD=(
  "$PY" -B "$EMBODIED_PATH/train_embodied_agent.py"
  --config-path "$EMBODIED_PATH/config"
  --config-name "$CFG"
  "runner.logger.log_path=$RUN_ROOT"
  "runner.resume_dir=$CKPT1"
  "runner.max_steps=2"
)
printf '%q ' "${RESUME_CMD[@]}" > "$RUN_ROOT/resume_command.txt"
printf '\n' >> "$RUN_ROOT/resume_command.txt"
nohup "${RESUME_CMD[@]}" > "$RUN_ROOT/resume_driver.log" 2>&1 < /dev/null &
RESUME_PID=$!
printf '%s\n' "$RESUME_PID" > "$RUN_ROOT/resume.pid"

nohup "$PY" -B "$REPO/examples/embodiment/monitor_resources.py" \
  --pid "$RESUME_PID" \
  --out-dir "$RUN_ROOT/resource_monitor/resume" \
  --interval 2 \
  > "$RUN_ROOT/resource_monitor/resume/monitor.log" 2>&1 < /dev/null &
MONITOR_PID=$!
printf '%s\n' "$MONITOR_PID" > "$RUN_ROOT/resource_monitor/resume/monitor.pid"

{
  echo "resume_started_at=$(date --iso-8601=seconds)"
  echo "resume_pid=$RESUME_PID"
  echo "resume_monitor_pid=$MONITOR_PID"
  echo "resume_source=$CKPT1"
  echo "resume_target=$CKPT2"
} > "$RUN_ROOT/resume_provenance.txt"

echo "RESUME_PID=$RESUME_PID"
echo "MONITOR_PID=$MONITOR_PID"
echo "RUN_ROOT=$RUN_ROOT"
echo "RESUME_STARTED_AT=$(date --iso-8601=seconds)"
