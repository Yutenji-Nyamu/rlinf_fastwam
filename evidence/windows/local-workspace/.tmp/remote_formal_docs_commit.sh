set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = d664bf349b63b75f41d51c8295cb0a330780d783
kill -0 70062

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP15_20260728.md \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_PROGRESS_CURVES_STEP15_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_CURVES_STEP15_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP15_20260728.png

git diff --cached --check
echo "STAGED"
git diff --cached --name-status
test "$(git diff --cached --name-only | wc -l)" -eq 8

git commit -m "docs(dsrl): record formal step 15 status"
echo "COMMIT=$(git rev-parse HEAD)"
git status --short
