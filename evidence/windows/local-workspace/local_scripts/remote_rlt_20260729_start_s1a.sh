set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
DATA_ROOT=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-contract-ep0-v1
MODEL_ROOT=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
STATS_PATH="$MODEL_ROOT/physical-intelligence/robotwin/norm_stats.json"
RUN_ROOT=/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1
S1A_NAME=robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1
RUNTIME_ROOT="$EVIDENCE_ROOT/s1a_runtime"
MONITOR_SCRIPT=/root/autodl-tmp/tmp/rlt_stage1_resource_monitor_20260729.sh

cd "$RLT_ROOT"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test -z "$(git status --short)"
test -d "$DATA_ROOT"
test -f "$STATS_PATH"
test -f "$MONITOR_SCRIPT"
test ! -e "$RUN_ROOT/s1a/$S1A_NAME"
test ! -e "$RUNTIME_ROOT"
if pgrep -af 'train_vla_sft|ray::|raylet|gcs_server' | grep -v -E 'pgrep -af|start_s1a' >/dev/null; then
  printf '%s\n' 'refusing to start: relevant process already exists'
  pgrep -af 'train_vla_sft|ray::|raylet|gcs_server' || true
  exit 3
fi

mkdir -p "$RUNTIME_ROOT"
cat > "$RUNTIME_ROOT/run_foreground.sh" <<EOF
set +e
cd "$RLT_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export ROBOTWIN_RLT_CLEAN50_PATH="$DATA_ROOT"
export ROBOTWIN_PI0_BASE_PATH="$MODEL_ROOT"
export ROBOTWIN_PI0_NORM_STATS_PATH="$STATS_PATH"
date -Is > "$RUNTIME_ROOT/started_at.txt"
timeout --signal=TERM --kill-after=60s 1800s \
  "$PYTHON_BIN" -B examples/sft/train_vla_sft.py \
    --config-path "$RLT_ROOT/examples/sft/config" \
    --config-name robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke \
    "runner.logger.log_path=$RUN_ROOT/s1a" \
    "runner.logger.experiment_name=$S1A_NAME"
rc=\$?
printf '%s\n' "\$rc" > "$RUNTIME_ROOT/exit_code.txt"
date -Is > "$RUNTIME_ROOT/finished_at.txt"
exit "\$rc"
EOF
chmod 700 "$RUNTIME_ROOT/run_foreground.sh"

nohup "$RUNTIME_ROOT/run_foreground.sh" \
  > "$RUNTIME_ROOT/driver.log" \
  2>&1 \
  < /dev/null &
driver_pid=$!
printf '%s\n' "$driver_pid" > "$RUNTIME_ROOT/driver_pid.txt"

nohup bash "$MONITOR_SCRIPT" \
  "$driver_pid" \
  "$RUNTIME_ROOT/resources.csv" \
  > "$RUNTIME_ROOT/monitor.log" \
  2>&1 \
  < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" > "$RUNTIME_ROOT/monitor_pid.txt"

printf 'DRIVER_PID=%s\nMONITOR_PID=%s\nRUNTIME_ROOT=%s\n' \
  "$driver_pid" "$monitor_pid" "$RUNTIME_ROOT"
