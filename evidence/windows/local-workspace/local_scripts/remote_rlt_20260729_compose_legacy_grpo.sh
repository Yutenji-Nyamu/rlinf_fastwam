set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_pre_smoke_20260729
OUTPUT_PATH="$EVIDENCE_ROOT/legacy_robotwin_pi0_grpo_fastwam_resolved.yaml"

cd "$RLT_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1
export REPO_PATH="$RLT_ROOT"
export EMBODIED_PATH="$RLT_ROOT/examples/embodiment"

"$PYTHON_BIN" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$RLT_ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_fastwam_a800_2gpu \
  --cfg job \
  --resolve >"$OUTPUT_PATH"

wc -l "$OUTPUT_PATH"
sha256sum "$OUTPUT_PATH"
