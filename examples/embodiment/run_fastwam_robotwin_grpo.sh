#!/usr/bin/env bash
set -euo pipefail

# Reproducible launcher for the verified AutoDL layout. It selects the one-step
# smoke or the 100-step baseline, then starts the standard RLinf entrypoint and
# the same sidecar resource monitor used by the earlier pi0 PPO/GRPO runs.
ROOT="${ROOT:-/root/autodl-tmp}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${FASTWAM_RLINF_ENV:-$ROOT/conda/envs/FastWAM-RLinf}"
PYTHON="$ENV/bin/python"
test -x "$PYTHON" || {
  echo "STOP: missing joint-environment Python: $PYTHON" >&2
  exit 1
}
export PATH="$ENV/bin:$PATH"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
unset HF_ENDPOINT DIFFSYNTH_SKIP_DOWNLOAD

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

export ROBOTWIN_PATH="${ROBOTWIN_PATH:-$ROOT/RoboTwin_RLinf}"
export ROBOT_PLATFORM=ALOHA
export FASTWAM_CONFIG_DIR="${FASTWAM_CONFIG_DIR:-$ROOT/FastWAM/configs}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/cache/modelscope}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$ROOT/models/fastwam/diffsynth}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export EMBODIED_PATH="$REPO/examples/embodiment"
export REPO_PATH="$REPO"
export PYTHONPATH="$REPO:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"

mode="${1:-smoke}"
case "$mode" in
  smoke)
    default_config=robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke
    ;;
  train)
    default_config=robotwin_adjust_bottle_grpo_fastwam_a800_2gpu
    ;;
  *)
    echo "usage: $0 [smoke|train] [config-name]" >&2
    exit 2
    ;;
esac

config="${2:-$default_config}"
test -f "$EMBODIED_PATH/config/$config.yaml" || {
  echo "STOP: missing config: $EMBODIED_PATH/config/$config.yaml" >&2
  exit 1
}

if pgrep -af '[t]rain_embodied_agent.py' >/dev/null; then
  echo "STOP: another embodied training driver is already running" >&2
  pgrep -af '[t]rain_embodied_agent.py' >&2 || true
  exit 1
fi

run_name="$(date +'%Y%m%d_%H%M%S')-${config}"
log_dir="$REPO/logs/$run_name"
run_log="$log_dir/run_embodiment.log"
monitor_dir="$log_dir/resource_monitor"
monitor_log="$monitor_dir/monitor.log"
src="$EMBODIED_PATH/train_embodied_agent.py"

mkdir -p "$monitor_dir"

cmd=(
  "$PYTHON"
  "$src"
  --config-path "$EMBODIED_PATH/config/"
  --config-name "$config"
  "runner.logger.log_path=$log_dir"
  "runner.logger.experiment_name=$config"
)

printf '%q ' "${cmd[@]}" > "$log_dir/command.txt"
printf '\n' >> "$log_dir/command.txt"

# Resolve the exact Hydra job before launching so each run is self-describing.
"$PYTHON" "$src" \
  --config-path "$EMBODIED_PATH/config/" \
  --config-name "$config" \
  --cfg job --resolve > "$log_dir/resolved_config.yaml"

cd "$REPO"
nohup "${cmd[@]}" > "$run_log" 2>&1 &
train_pid=$!
printf '%s\n' "$train_pid" > "$log_dir/train.pid"

nohup "$PYTHON" "$EMBODIED_PATH/monitor_resources.py" \
  --pid "$train_pid" \
  --out-dir "$monitor_dir" \
  --interval 2 \
  > "$monitor_log" 2>&1 &
monitor_pid=$!
printf '%s\n' "$monitor_pid" > "$log_dir/monitor.pid"

echo "MODE=$mode"
echo "CONFIG=$config"
echo "TRAIN_PID=$train_pid"
echo "MONITOR_PID=$monitor_pid"
echo "LOG_DIR=$log_dir"
echo "RUN_LOG=$run_log"
echo "RESOURCE_CSV=$monitor_dir/resources.csv"
echo "PEAK_FILE=$monitor_dir/peak.txt"
echo "Follow with: tail -f '$run_log'"
