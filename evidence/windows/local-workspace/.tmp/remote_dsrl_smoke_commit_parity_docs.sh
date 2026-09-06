set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
BRANCH=codex/dsrl-pi0-robotwin
UPLOAD="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1/docs_upload"
DOC_ROOT="$REPO/docs/rlinf-robotwin-pi0-traditional-rl"
PATHS=(
  docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/IMPLEMENTATION_LOG.md
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_EXECUTION_LOG_20260728.md
)

test "$(git -C "$REPO" rev-parse HEAD)" = 3c7d35d716cfd6964b15249e548a0aab99d4cb27
test "$(git -C "$REPO" rev-parse '@{upstream}')" = 3c7d35d716cfd6964b15249e548a0aab99d4cb27
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
echo "8714ab6c1b83a20e118366b70de77cfc40901b0f31b9190c2bb96feef975849f  $UPLOAD/00_INDEX_AND_IMPLEMENTATION_PLAN.md" | sha256sum -c -
echo "9735a5cf9e315c7b3677caff4f9822f98c064532ac15f29f025612ead26e953a  $UPLOAD/IMPLEMENTATION_LOG.md" | sha256sum -c -
echo "ff745e3c67867a636bb7ac52aff94f6ee1129d674862c9b6f637223412a2cb99  $UPLOAD/SMOKE_EXECUTION_LOG_20260728.md" | sha256sum -c -

install -m 0644 "$UPLOAD/00_INDEX_AND_IMPLEMENTATION_PLAN.md" "$DOC_ROOT/00_INDEX_AND_IMPLEMENTATION_PLAN.md"
install -m 0644 "$UPLOAD/IMPLEMENTATION_LOG.md" "$DOC_ROOT/evidence/IMPLEMENTATION_LOG.md"
install -m 0644 "$UPLOAD/SMOKE_EXECUTION_LOG_20260728.md" "$DOC_ROOT/evidence/SMOKE_EXECUTION_LOG_20260728.md"
test "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all | wc -l)" -eq 3
git -C "$REPO" diff --check
git -C "$REPO" add -- "${PATHS[@]}"
test -z "$(git -C "$REPO" diff --name-only)"
git -C "$REPO" diff --cached --check
git -C "$REPO" diff --cached --stat
git -C "$REPO" commit -s -m "docs(embodiment): record fixed-latent parity"
commit=$(git -C "$REPO" rev-parse HEAD)
timeout --signal=TERM --kill-after=5s 60s \
  git -C "$REPO" -c http.version=HTTP/1.1 push personal "$BRANCH"
test "$(git -C "$REPO" rev-parse '@{upstream}')" = "$commit"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
echo "PUSHED_HEAD=$commit"
echo "PARITY_DOCS_COMMIT_OK=1"
