set -euo pipefail
# Docs-only validation used before the step-20 report commit.

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = 95e6251841f6d7256ee2c13de053d4618e02e00e
test "$(git rev-parse '@{upstream}')" = 95e6251841f6d7256ee2c13de053d4618e02e00e
kill -0 70062

echo "STATUS"
git status --short
git diff --check

for image in \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP20_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP20_20260728.png \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP20_20260728.png
do
  test -s "$image"
  file "$image"
done

test "$(wc -l < docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STEP_TIMING_STEP20_20260728.csv)" -eq 21
grep -q 'FORMAL-010' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md
grep -q '总共有多少 step' \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP20_20260728.md

PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/RLinf/.venv/bin/python -B \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/build_dsrl_formal_step20_plots.py \
  --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/RLinf/.venv/bin/python -B \
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/build_dsrl_formal_resource_plot.py \
  --help >/dev/null

echo "VALIDATION=PASS"
