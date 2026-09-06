set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
BRANCH=codex/dsrl-pi0-robotwin
PATHS=(
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/IMPLEMENTATION_LOG.md
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_APPROVAL_20260728.md
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_EXECUTION_LOG_20260728.md
)

test "$(git -C "$REPO" branch --show-current)" = "$BRANCH"
test "$(git -C "$REPO" rev-parse HEAD)" = 2d942b714b004de9a7efdbd4a7e2efaac3ef6d01
test "$(git -C "$REPO" rev-parse '@{upstream}')" = 2d942b714b004de9a7efdbd4a7e2efaac3ef6d01
test "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all | wc -l)" -eq 4

git -C "$REPO" add -- "${PATHS[@]}"
test -z "$(git -C "$REPO" diff --name-only)"
test "$(git -C "$REPO" diff --cached --name-only | wc -l)" -eq 4
git -C "$REPO" diff --cached --check
git -C "$REPO" diff --cached --stat
git -C "$REPO" diff --cached --numstat

git -C "$REPO" commit -s -m "docs(embodiment): record DSRL smoke results"
commit=$(git -C "$REPO" rev-parse HEAD)
echo "DOCS_COMMIT=$commit"
git -C "$REPO" push personal "$BRANCH"
test "$(git -C "$REPO" rev-parse '@{upstream}')" = "$commit"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
echo "PUSHED_HEAD=$commit"
echo "DOCS_COMMIT_PUSH_OK=1"
