set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
stage=/root/autodl-tmp/qam_docs_sync_20260801_v4

test "$(git -C "$repo" rev-parse HEAD)" = 24cbc8d20d19161c46da9940b5731127530e911d
test -z "$(git -C "$repo" status --short)"

install -m 0644 "$stage/HANDOFF.md" "$repo/HANDOFF.md"
install -m 0644 "$stage/00_INDEX_AND_IMPLEMENTATION_PLAN.md" \
  "$repo/docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md"
install -m 0644 "$stage/IMPLEMENTATION_LOG.md" \
  "$repo/docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md"
install -m 0755 "$stage/qam_formal_resume100_to380_launch_20260801_v4.sh" \
  "$repo/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume100_to380_launch_20260801_v4.sh"

bash -n "$repo/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume100_to380_launch_20260801_v4.sh"
git -C "$repo" diff --check
git -C "$repo" status --short
git -C "$repo" diff --stat
