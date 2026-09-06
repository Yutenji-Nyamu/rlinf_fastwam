set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
ROBOTWIN=/root/autodl-tmp/RoboTwin_RLinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
CFG=robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1

test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/resource_monitor/fresh"

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

FRESH_CMD=(
  "$PY" -B "$EMBODIED_PATH/train_embodied_agent.py"
  --config-path "$EMBODIED_PATH/config"
  --config-name "$CFG"
  "runner.logger.log_path=$RUN_ROOT"
  "runner.resume_dir=null"
)
printf '%q ' "${FRESH_CMD[@]}" > "$RUN_ROOT/fresh_command.txt"
printf '\n' >> "$RUN_ROOT/fresh_command.txt"
nohup "${FRESH_CMD[@]}" > "$RUN_ROOT/fresh_driver.log" 2>&1 < /dev/null &
FRESH_PID=$!
printf '%s\n' "$FRESH_PID" > "$RUN_ROOT/fresh.pid"

nohup "$PY" -B "$REPO/examples/embodiment/monitor_resources.py" \
  --pid "$FRESH_PID" \
  --out-dir "$RUN_ROOT/resource_monitor/fresh" \
  --interval 2 \
  > "$RUN_ROOT/resource_monitor/fresh/monitor.log" 2>&1 < /dev/null &
MONITOR_PID=$!
printf '%s\n' "$MONITOR_PID" > "$RUN_ROOT/resource_monitor/fresh/monitor.pid"

echo "FRESH_PID=$FRESH_PID"
echo "MONITOR_PID=$MONITOR_PID"
echo "RUN_ROOT=$RUN_ROOT"
echo "FRESH_STARTED_AT=$(date --iso-8601=seconds)"
