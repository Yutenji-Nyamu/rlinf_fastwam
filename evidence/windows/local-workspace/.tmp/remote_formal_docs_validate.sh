set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
cd "$REPO"
kill -0 70062

echo "STATUS"
git status --short
echo "DIFF_STAT"
git diff --stat
git diff --check

test "$(sha256sum docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml | awk '{print $1}')" = \
  e99c212d1743e285dcda23cb129e2ed96545cceb36bebe772ae69a693b9df595

for image in \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_PROGRESS_CURVES_STEP15_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_CURVES_STEP15_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP15_20260728.png
do
  test -s "$image"
  file "$image"
done

grep -q '最新完整记录是 global' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP15_20260728.md
grep -q 'FORMAL-008' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
echo "VALIDATION=PASS"
