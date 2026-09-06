set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_pre_smoke_20260729
STDOUT_PATH="$EVIDENCE_ROOT/robotwin_rlt_prefix_probe_base.json"
STDERR_PATH="$EVIDENCE_ROOT/robotwin_rlt_prefix_probe_base.stderr"
PID_PATH="$EVIDENCE_ROOT/robotwin_rlt_prefix_probe_base.pid"

cd "$RLT_ROOT"
mkdir -p "$EVIDENCE_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0

nohup "$PYTHON_BIN" -B \
  toolkits/rlt/probe_robotwin_rlt_prefix_contract.py \
  --device cuda:0 \
  --expected-hidden-width 2048 \
  --expected-image-tokens 768 \
  >"$STDOUT_PATH" 2>"$STDERR_PATH" &
probe_pid=$!
printf '%s\n' "$probe_pid" >"$PID_PATH"
printf 'PID=%s\nSTDOUT=%s\nSTDERR=%s\n' \
  "$probe_pid" "$STDOUT_PATH" "$STDERR_PATH"
