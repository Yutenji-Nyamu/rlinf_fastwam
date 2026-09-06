set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
echo "PARENT_PROXY_BEFORE=$(env | grep -icE '^(http|https|all)_proxy=' || true)"
(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u
  timeout 15 git ls-remote personal refs/heads/codex/qam-pi0-robotwin
  GIT_TERMINAL_PROMPT=0 timeout 60 \
    git push personal HEAD:refs/heads/codex/qam-pi0-robotwin
  timeout 15 git ls-remote personal refs/heads/codex/qam-pi0-robotwin
)
echo "PARENT_PROXY_AFTER=$(env | grep -icE '^(http|https|all)_proxy=' || true)"
echo "HEAD=$(git rev-parse HEAD)"
echo "STATUS=$(git status --porcelain | wc -l)"
echo "AHEAD_BEHIND=$(git rev-list --left-right --count '@{upstream}...HEAD')"

