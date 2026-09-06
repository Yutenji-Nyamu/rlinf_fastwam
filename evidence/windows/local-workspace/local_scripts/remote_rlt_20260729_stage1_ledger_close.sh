set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
cd "$RLT_ROOT"

test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = c22ba19af0b6dfd130e289f02efd3a42ce5e938f
git diff --cached --quiet

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md

test "$(git diff --cached --name-only | wc -l)" = 2
git diff --cached --name-only | grep -Fx HANDOFF.md
git diff --cached --name-only | grep -Fx \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
git diff --cached --check
test -z "$(git diff --name-only)"
test -z "$(git ls-files --others --exclude-standard)"

git commit -m "docs(rlt): close Stage 1 smoke ledger"
close_commit=$(git rev-parse HEAD)
timeout --signal=TERM --kill-after=10s 90s \
  git push personal codex/rlt-pi0-robotwin

printf 'CLOSE_COMMIT=%s\n' "$close_commit"
printf '%s\n' 'STATUS'
git status --short
printf '%s\n' 'AHEAD_BEHIND'
git rev-list --left-right --count HEAD...@{upstream}
printf '%s\n' 'REMOTE_HEAD'
git ls-remote personal refs/heads/codex/rlt-pi0-robotwin
