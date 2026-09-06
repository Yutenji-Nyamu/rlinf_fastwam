set -u

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
echo "PROXY_BEFORE"
env | grep -iE '^(http|https|all)_proxy=' || true
echo "HEAD=$(git rev-parse HEAD)"
echo "STATUS=$(git status --porcelain | wc -l)"
echo "AHEAD_BEHIND=$(git rev-list --left-right --count '@{upstream}...HEAD')"
timeout 15 git ls-remote personal refs/heads/codex/qam-pi0-robotwin
echo "LS_REMOTE_EXIT=$?"

