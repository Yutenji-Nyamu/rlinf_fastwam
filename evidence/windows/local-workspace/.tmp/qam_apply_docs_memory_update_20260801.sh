set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
stage=/root/autodl-tmp/qam_docs_update_20260801
archive=/root/autodl-tmp/qam_docs_memory_update_20260801.tar

test "$(git -C "$repo" rev-parse HEAD)" = 9e2abc04e8c178575d9b800154d69b9123e73ecb
test -z "$(git -C "$repo" status --short)"
tar -xf "$archive" -C "$stage"
cp "$stage/PROJECT_CONTEXT.md" "$repo/PROJECT_CONTEXT.md"
cp "$stage/HANDOFF.md" "$repo/HANDOFF.md"
cp "$stage/docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md" \
  "$repo/docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md"
cp "$stage/docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md" \
  "$repo/docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md"
cp "$stage/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume25_to100_launch_20260801_v3.sh" \
  "$repo/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume25_to100_launch_20260801_v3.sh"
bash -n "$repo/docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume25_to100_launch_20260801_v3.sh"
git -C "$repo" diff --check
git -C "$repo" status --short
git -C "$repo" diff --stat
