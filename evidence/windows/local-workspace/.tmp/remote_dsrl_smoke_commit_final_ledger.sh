set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
BRANCH=codex/dsrl-pi0-robotwin
UPLOAD="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1/docs_upload"
EVIDENCE="$REPO/docs/rlinf-robotwin-pi0-traditional-rl/evidence"
PATHS=(
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/IMPLEMENTATION_LOG.md
  docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_EXECUTION_LOG_20260728.md
)

test "$(git -C "$REPO" rev-parse HEAD)" = ff0d8d2270aa5ec4f0934c997f53118460bf8152
test "$(git -C "$REPO" rev-parse '@{upstream}')" = ff0d8d2270aa5ec4f0934c997f53118460bf8152
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
echo "8013a63c16f5ca55cbde29e197385ea0642f690f7c76fae11ef853fea3c94750  $UPLOAD/IMPLEMENTATION_LOG.md" | sha256sum -c -
echo "1b8c2d6c365ed4ae842337ab177a11caff8ea40558aa4384d4afc0ddcf1560c8  $UPLOAD/SMOKE_EXECUTION_LOG_20260728.md" | sha256sum -c -

install -m 0644 "$UPLOAD/IMPLEMENTATION_LOG.md" "$EVIDENCE/IMPLEMENTATION_LOG.md"
install -m 0644 "$UPLOAD/SMOKE_EXECUTION_LOG_20260728.md" "$EVIDENCE/SMOKE_EXECUTION_LOG_20260728.md"
test "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all | wc -l)" -eq 2
git -C "$REPO" diff --check
git -C "$REPO" add -- "${PATHS[@]}"
test -z "$(git -C "$REPO" diff --name-only)"
git -C "$REPO" diff --cached --check
git -C "$REPO" diff --cached --stat
git -C "$REPO" commit -s -m "docs(embodiment): close DSRL smoke ledger"
commit=$(git -C "$REPO" rev-parse HEAD)
timeout --signal=TERM --kill-after=5s 60s \
  git -C "$REPO" -c http.version=HTTP/1.1 push personal "$BRANCH"
test "$(git -C "$REPO" rev-parse '@{upstream}')" = "$commit"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
echo "PUSHED_HEAD=$commit"
echo "FINAL_LEDGER_COMMIT_OK=1"
