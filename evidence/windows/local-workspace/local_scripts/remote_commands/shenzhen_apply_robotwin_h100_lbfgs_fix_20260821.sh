#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/robotwin-native/RoboTwin
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
PATCH="$RUN/robotwin_h100_cuda121_unfused_lbfgs.patch"
LOG="$RUN/03_apply_h100_lbfgs_fix.log"

exec > >(tee "$LOG") 2>&1
cd "$ROOT"

test "$(git rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
git diff --quiet -- envs/robot/planner.py
printf '%s  %s\n' 655D520E7A2208A9910279EE04A84C8B719836B32EE00992BC1A274377645091 "$PATCH" | sha256sum --check -
git apply --check "$PATCH"
git apply "$PATCH"

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin
python -m py_compile envs/robot/planner.py
git diff --check -- envs/robot/planner.py
git status --short -- envs/robot/planner.py
git diff -- envs/robot/planner.py
