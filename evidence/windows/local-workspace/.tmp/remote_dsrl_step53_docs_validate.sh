set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=b01661e8a6b3ca1b883fb61d4ade9a467ffd84b5

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
kill -0 "$(cat "$RUN/formal.pid")"

expected_paths=$(cat <<'EOF'
HANDOFF.md
docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP53_20260728_WIDE.png
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP53_20260728_WIDE.png
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP53_20260728.md
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP53_20260728_WIDE.png
docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
EOF
)
actual_paths=$(git status --porcelain | sed 's/^...//' | sort)
test "$actual_paths" = "$expected_paths"
git diff --check

for image in \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP53_20260728_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP53_20260728_WIDE.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP53_20260728_WIDE.png
do
  test -s "$image"
  file "$image" | grep -q '2430 x 1440'
done

grep -q '10/12 = 83.3%' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP53_20260728.md
grep -q 'FORMAL-013' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
grep -q 'step 53' HANDOFF.md
grep -q 'step 53' \
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md

echo "VALIDATION=PASS"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
