set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=acc7c14b93aec8eb2f2e8f32e4072be3957b761b

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
! kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null

expected_paths=$(cat <<'EOF'
HANDOFF.md
docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_FINAL_STEP198_20260729_WIDE.png
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_FINAL_STEP198_20260729_WIDE.png
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STEP_TIMING_FINAL_FLUSH_STEP197_20260729.csv
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_FINAL_STEP198_20260729_WIDE.png
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TENSORBOARD_METRICS_FINAL_FLUSH_STEP197_20260729.json
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
EOF
)
actual_paths=$(git status --porcelain | sed 's/^...//' | sort)
if test "$actual_paths" != "$expected_paths"; then
  echo "UNEXPECTED_DIRTY_PATHS"
  printf '%s\n' "$actual_paths"
  exit 11
fi
git diff --check

for image in \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_FINAL_STEP198_20260729_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_FINAL_STEP198_20260729_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_FINAL_STEP198_20260729_WIDE.png
do
  test -s "$image"
  file "$image" | grep -q '2430 x 1440'
done

grep -q 'step 198/650' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md
grep -q 'FORMAL-015' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
grep -q 'step 198' HANDOFF.md
grep -q 'step 198' \
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md

echo "VALIDATION=PASS"
git status --short
