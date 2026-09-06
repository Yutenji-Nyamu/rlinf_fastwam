set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
UPLOAD="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1/docs_upload"
DOC_ROOT="$REPO/docs/rlinf-robotwin-pi0-traditional-rl"

test "$(git -C "$REPO" branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git -C "$REPO" rev-parse HEAD)" = 2d942b714b004de9a7efdbd4a7e2efaac3ef6d01
test "$(git -C "$REPO" rev-parse '@{upstream}')" = 2d942b714b004de9a7efdbd4a7e2efaac3ef6d01
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"

echo "UPLOAD_HASHES"
echo "9948a118dd4643e4ba80282b3932ea2f9aa01d3cb42679afa75424cbdb0de1a9  $UPLOAD/00_INDEX_AND_IMPLEMENTATION_PLAN.md" | sha256sum -c -
echo "24c8dab11fdd69380d76347024042d89121fb91f4bbcc2534eb8ca51ea4f3e41  $UPLOAD/IMPLEMENTATION_LOG.md" | sha256sum -c -
echo "125af91b8809f8c188e6b24920b15b27586be2b9758197a02077408d9672ff44  $UPLOAD/SMOKE_APPROVAL_20260728.md" | sha256sum -c -
echo "4bab08b35b16e8bc2f0e8321043ddaadf3121f6225f2fbed8b373bc6698da601  $UPLOAD/SMOKE_EXECUTION_LOG_20260728.md" | sha256sum -c -

install -m 0644 "$UPLOAD/00_INDEX_AND_IMPLEMENTATION_PLAN.md" "$DOC_ROOT/00_INDEX_AND_IMPLEMENTATION_PLAN.md"
install -m 0644 "$UPLOAD/IMPLEMENTATION_LOG.md" "$DOC_ROOT/evidence/IMPLEMENTATION_LOG.md"
install -m 0644 "$UPLOAD/SMOKE_APPROVAL_20260728.md" "$DOC_ROOT/evidence/SMOKE_APPROVAL_20260728.md"
install -m 0644 "$UPLOAD/SMOKE_EXECUTION_LOG_20260728.md" "$DOC_ROOT/evidence/SMOKE_EXECUTION_LOG_20260728.md"

echo "TARGET_HASHES"
sha256sum \
  "$DOC_ROOT/00_INDEX_AND_IMPLEMENTATION_PLAN.md" \
  "$DOC_ROOT/evidence/IMPLEMENTATION_LOG.md" \
  "$DOC_ROOT/evidence/SMOKE_APPROVAL_20260728.md" \
  "$DOC_ROOT/evidence/SMOKE_EXECUTION_LOG_20260728.md"

git -C "$REPO" diff --check
status=$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)
printf "%s\n" "$status"
test "$(printf "%s\n" "$status" | wc -l)" -eq 4
printf "%s\n" "$status" | grep -F "docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md"
printf "%s\n" "$status" | grep -F "docs/rlinf-robotwin-pi0-traditional-rl/evidence/IMPLEMENTATION_LOG.md"
printf "%s\n" "$status" | grep -F "docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_APPROVAL_20260728.md"
printf "%s\n" "$status" | grep -F "docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_EXECUTION_LOG_20260728.md"

git -C "$REPO" diff --stat
git -C "$REPO" diff --numstat
echo "DOCS_APPLY_OK=1"
