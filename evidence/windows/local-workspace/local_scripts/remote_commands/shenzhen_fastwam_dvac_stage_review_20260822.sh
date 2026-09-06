#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
cd "$WT"
test "$(git rev-parse HEAD)" = 7faa71108368fbb3b6885649f112af607427a2d4
test "$(git branch --show-current)" = codex/sz-fastwam-dvac-observe

git add -- \
  configs/sim_robotwin.yaml \
  experiments/robotwin/eval_robotwin_single.py \
  experiments/robotwin/fastwam_policy/deploy_policy.py \
  experiments/robotwin/fastwam_policy/deploy_policy.yml \
  experiments/robotwin/fastwam_policy/dvac_telemetry.py \
  src/fastwam/models/wan22/fastwam.py
git add -f -- tests/test_fastwam_dvac_telemetry.py

echo "TIME_UTC=$(date -u +%FT%TZ)"
echo STATUS_BEGIN
git status --short
echo STATUS_END
echo CACHED_STAT_BEGIN
git diff --cached --stat
echo CACHED_STAT_END
echo CACHED_NUMSTAT_BEGIN
git diff --cached --numstat
echo CACHED_NUMSTAT_END
git diff --cached --check

echo CONTRACT_LINES_BEGIN
grep -nE \
  'return_action_denoising_trace|action_denoising_trace|device="cpu"|FileExistsError|begin_episode|finalize_episode|dvac_telemetry|eval_success|record_action' \
  src/fastwam/models/wan22/fastwam.py \
  experiments/robotwin/fastwam_policy/deploy_policy.py \
  experiments/robotwin/fastwam_policy/dvac_telemetry.py \
  experiments/robotwin/eval_robotwin_single.py \
  configs/sim_robotwin.yaml \
  tests/test_fastwam_dvac_telemetry.py
echo CONTRACT_LINES_END
echo FASTWAM_DVAC_STAGE_REVIEW_OK
