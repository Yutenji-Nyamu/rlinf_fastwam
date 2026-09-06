set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=b01661e8a6b3ca1b883fb61d4ade9a467ffd84b5

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
kill -0 "$(cat "$RUN/formal.pid")"

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP53_20260728.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP53_20260728_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP53_20260728_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP53_20260728_WIDE.png

git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 7
echo "STAGED_BEGIN"
git diff --cached --name-status
echo "STAGED_END"

git commit -m "docs(dsrl): refresh formal step 53 report"
timeout --signal=TERM --kill-after=5s 45s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin

echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
kill -0 "$(cat "$RUN/formal.pid")"
echo "DRIVER_ALIVE=1"
echo "LAST_STEP=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"
