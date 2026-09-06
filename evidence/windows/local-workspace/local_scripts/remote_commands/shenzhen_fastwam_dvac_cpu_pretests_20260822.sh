#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
PYTHON=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
cd "$WT"

echo "TIME_UTC=$(date -u +%FT%TZ)"
echo "HEAD=$(git rev-parse HEAD) BRANCH=$(git branch --show-current)"
test "$(git rev-parse HEAD)" = 7faa71108368fbb3b6885649f112af607427a2d4
test "$(git branch --show-current)" = codex/sz-fastwam-dvac-observe

echo STATUS_BEGIN
git status --short
echo STATUS_END
echo DIFF_STAT_BEGIN
git diff --stat
echo DIFF_STAT_END
git diff --check

test -x "$PYTHON"
"$PYTHON" -V
"$PYTHON" -m py_compile \
  src/fastwam/models/wan22/fastwam.py \
  experiments/robotwin/fastwam_policy/deploy_policy.py \
  experiments/robotwin/fastwam_policy/dvac_telemetry.py \
  experiments/robotwin/eval_robotwin_single.py \
  tests/test_fastwam_dvac_telemetry.py

"$PYTHON" tests/test_fastwam_dvac_telemetry.py

"$PYTHON" - <<'PY'
import inspect

from omegaconf import OmegaConf

from experiments.robotwin.fastwam_policy.dvac_telemetry import compute_z_endpoint
from fastwam.models.wan22.fastwam import FastWAM

cfg = OmegaConf.load("configs/sim_robotwin.yaml")
assert cfg.EVALUATION.dvac_telemetry.enabled is False
signature = inspect.signature(FastWAM.infer_action)
assert signature.parameters["return_action_denoising_trace"].default is False
assert callable(compute_z_endpoint)
print("FASTWAM_DVAC_DEFAULT_OFF_IMPORT_OK")
PY

echo FINAL_STATUS_BEGIN
git status --short
echo FINAL_STATUS_END
echo FASTWAM_DVAC_CPU_PRETESTS_OK
