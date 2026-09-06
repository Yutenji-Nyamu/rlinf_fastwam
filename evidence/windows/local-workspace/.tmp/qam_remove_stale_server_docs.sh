set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
root="$repo/docs/rlinf-robotwin-pi0-qam"
test -f "$root/00_INDEX_AND_IMPLEMENTATION_PLAN.md"
test -f "$root/01_CONTEXT_AND_SOURCE_MAP.md"
test -f "$root/02_METHOD_AND_PORT_DECISION_GUIDE.md"
test -f "$root/evidence/IMPLEMENTATION_LOG.md"
rm -- \
  "$root/00_INDEX_AND_IMPLEMENTATION_PLAN.md" \
  "$root/01_CONTEXT_AND_SOURCE_MAP.md" \
  "$root/02_METHOD_AND_PORT_DECISION_GUIDE.md" \
  "$root/evidence/IMPLEMENTATION_LOG.md"
rmdir "$root/evidence"
rmdir "$root"
test -z "$(git -C "$repo" status --short)"
printf 'REMOVED_STALE_SERVER_DOC_DUPLICATE=%s\n' "$root"
