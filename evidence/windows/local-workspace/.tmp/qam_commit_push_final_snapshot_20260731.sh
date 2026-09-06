set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
test -z "$(
  git status --short |
    grep -v '^ M docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md$' \
    || true
)"
git diff --check
git add -- docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
git diff --cached --check
git commit -m "docs(qam): record final pre-smoke snapshot"
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

