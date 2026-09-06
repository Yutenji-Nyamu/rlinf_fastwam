set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=acc7c14b93aec8eb2f2e8f32e4072be3957b761b

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
! kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_FINAL_STEP198_20260729_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_FINAL_STEP198_20260729_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_FINAL_STEP198_20260729_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TENSORBOARD_METRICS_FINAL_FLUSH_STEP197_20260729.json \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STEP_TIMING_FINAL_FLUSH_STEP197_20260729.csv

git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 9
git commit -m "docs(dsrl): close formal run at step 198"

timeout --signal=TERM --kill-after=5s 55s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin

head_now=$(git rev-parse HEAD)
test "$(git rev-parse '@{upstream}')" = "$head_now"
test -z "$(git status --porcelain)"
! kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null

echo "HEAD=$head_now"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS=CLEAN"
echo "DRIVER_ALIVE=0"
