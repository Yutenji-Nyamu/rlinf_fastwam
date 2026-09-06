set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
git -C "$repo" add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md \
  docs/rlinf-robotwin-pi0-qam/evidence/QAM_FORMAL_STOP247_CLOSEOUT_20260801.md
git -C "$repo" diff --cached --check
git -C "$repo" diff --cached --stat
git -C "$repo" commit -m 'docs(qam): close formal run at cycle 247'
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
