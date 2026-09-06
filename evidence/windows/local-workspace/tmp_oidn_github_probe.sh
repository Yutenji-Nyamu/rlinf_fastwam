set -u
command -v gh || true
gh auth status 2>&1 || true
gh api repos/Yutenji-Nyamu/RoboTwin --jq '.full_name + " private=" + (.private|tostring)' 2>&1 || true
GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10' git ls-remote git@github.com:Yutenji-Nyamu/RoboTwin.git HEAD 2>&1 || true
