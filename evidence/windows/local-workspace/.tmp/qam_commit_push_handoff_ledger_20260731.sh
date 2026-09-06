set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
test "$(git rev-parse HEAD)" = 3e7f26eb0cc38cc2f44e4145af480a71a2948262

git status --short
git diff --check
git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs(qam): record disk and publishing audit"
test -z "$(git status --short)"

echo "PARENT_PROXY_BEFORE=$(env | grep -icE '^(http|https|all)_proxy=' || true)"
(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u
  GIT_TERMINAL_PROMPT=0 timeout 60 \
    git push personal HEAD:refs/heads/codex/qam-pi0-robotwin
  timeout 15 git ls-remote personal refs/heads/codex/qam-pi0-robotwin
)
echo "PARENT_PROXY_AFTER=$(env | grep -icE '^(http|https|all)_proxy=' || true)"
echo "HEAD=$(git rev-parse HEAD)"
echo "STATUS=$(git status --porcelain | wc -l)"
echo "AHEAD_BEHIND=$(git rev-list --left-right --count '@{upstream}...HEAD')"

