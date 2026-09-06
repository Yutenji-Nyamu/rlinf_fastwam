set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test "$(git rev-parse HEAD)" = 9e2abc04e8c178575d9b800154d69b9123e73ecb
git diff --check
git add -- \
  PROJECT_CONTEXT.md \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume25_to100_launch_20260801_v3.sh
git diff --cached --check
git commit -m "docs(qam): record nonblocking formal restart"
test -z "$(git status --short)"
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
