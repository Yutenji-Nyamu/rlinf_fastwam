set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume100_to380_launch_20260801_v4.sh
git diff --cached --check
git diff --cached --stat
git commit -m "docs(qam): record resume to cycle 380"
printf 'head='
git rev-parse HEAD
git status --short
