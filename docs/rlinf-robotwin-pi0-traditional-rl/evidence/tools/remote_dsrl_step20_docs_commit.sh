set -euo pipefail
# Bounded docs-only commit/push for the step-20 formal training report.

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=95e6251841f6d7256ee2c13de053d4618e02e00e

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
kill -0 "$(cat "$RUN/formal.pid")"

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP20_20260728.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP20_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP20_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP20_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STEP_TIMING_STEP20_20260728.csv \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/build_dsrl_formal_step20_plots.py \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/build_dsrl_formal_resource_plot.py \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/remote_dsrl_metrics_refresh_snapshot.sh \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/remote_dsrl_step20_docs_validate.sh \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/remote_dsrl_step20_docs_commit.sh

git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 13
echo "STAGED_BEGIN"
git diff --cached --name-status
echo "STAGED_END"

git commit -m "docs(dsrl): report formal step 20 metrics"
timeout --signal=TERM --kill-after=5s 45s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin

echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
kill -0 "$(cat "$RUN/formal.pid")"
echo "DRIVER_ALIVE=1"
