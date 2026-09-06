set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
BRANCH=codex/dsrl-pi0-robotwin
REL=docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md
UPLOAD="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1/docs_upload/00_INDEX_AND_IMPLEMENTATION_PLAN.md"

test "$(git -C "$REPO" branch --show-current)" = "$BRANCH"
test "$(git -C "$REPO" rev-parse HEAD)" = 0346966d0b11225e3fb3c49d7a990bc5479dec9c
test "$(git -C "$REPO" rev-parse '@{upstream}')" = 0346966d0b11225e3fb3c49d7a990bc5479dec9c
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
echo "90f225b772b342979281a70a65b6f23cf716d7451887beb6f69dbe9020d65f51  $UPLOAD" | sha256sum -c -

install -m 0644 "$UPLOAD" "$REPO/$REL"
test "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all | wc -l)" -eq 1
git -C "$REPO" diff --check
git -C "$REPO" diff -- "$REL"
git -C "$REPO" add -- "$REL"
git -C "$REPO" diff --cached --check
git -C "$REPO" commit -s -m "docs(embodiment): clarify DSRL smoke provenance"
commit=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" push personal "$BRANCH"
test "$(git -C "$REPO" rev-parse '@{upstream}')" = "$commit"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
echo "PUSHED_HEAD=$commit"
echo "PLAN_PROVENANCE_FIX_OK=1"
