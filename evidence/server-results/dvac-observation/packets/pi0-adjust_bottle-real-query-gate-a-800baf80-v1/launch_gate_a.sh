#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
PACKET=/data/chenyiteng/results/dvac-observation/packets/pi0-adjust_bottle-real-query-gate-a-800baf80-v1
OUTPUT=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-real-query-gate-a-800baf80-v1
EXPECTED_HEAD=800baf80d6eab64169cf0e691eb04a681a093ee9
EXPECTED_BRANCH=codex/sz-current-pi0-dvac-observe

test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$WT" branch --show-current)" = "$EXPECTED_BRANCH"
test -z "$(git -C "$WT" status --porcelain=v1)"
test "$(git -C "$WT" rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test -x "$PY"
test -d "$MODEL"
test -f "$PACKET/resolved_config.yaml"
test ! -e "$OUTPUT"

export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export PYTHONPATH="$WT"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1

cd "$WT"
exec /usr/bin/time -v timeout --signal=INT --kill-after=120s 900s   "$PY" toolkits/probe_pi0_dvac_real_parity.py run   --resolved-config "$PACKET/resolved_config.yaml"   --model-path "$MODEL"   --expected-head "$EXPECTED_HEAD"   --reset-state-id 100100052   --inference-seed 0   --output-dir "$OUTPUT"
