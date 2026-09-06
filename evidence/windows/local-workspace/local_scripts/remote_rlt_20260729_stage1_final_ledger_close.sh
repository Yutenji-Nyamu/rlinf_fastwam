set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
LEDGER=docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
cd "$RLT_ROOT"

test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = 12f5a590a83c610071a879dfe62e41c98e1de0c8
git diff --cached --quiet
git add -- "$LEDGER"
test "$(git diff --cached --name-only)" = "$LEDGER"
git diff --cached --check
test -z "$(git diff --name-only)"
test -z "$(git ls-files --others --exclude-standard)"

git commit -m "docs(rlt): record resume contract verification"
final_commit=$(git rev-parse HEAD)
timeout --signal=TERM --kill-after=10s 90s \
  git push personal codex/rlt-pi0-robotwin

printf 'FINAL_COMMIT=%s\n' "$final_commit"
git status --short
git rev-list --left-right --count HEAD...@{upstream}
git ls-remote personal refs/heads/codex/rlt-pi0-robotwin
