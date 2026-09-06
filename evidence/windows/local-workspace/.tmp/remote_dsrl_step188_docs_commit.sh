set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=50ebc6780435b677fa507286e6252559fd6b9c79
EXPECTED_UPSTREAM=b01661e8a6b3ca1b883fb61d4ade9a467ffd84b5

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_UPSTREAM"
kill -0 "$(cat "$RUN/formal.pid")"

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP188_20260729.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP188_20260729_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP188_20260729_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP188_20260729_WIDE.png

git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 7
echo "STAGED_BEGIN"
git diff --cached --name-status
echo "STAGED_END"

git commit -m "docs(dsrl): refresh formal step 188 report"

echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
kill -0 "$(cat "$RUN/formal.pid")"
echo "DRIVER_ALIVE=1"
echo "LAST_STEP=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"

