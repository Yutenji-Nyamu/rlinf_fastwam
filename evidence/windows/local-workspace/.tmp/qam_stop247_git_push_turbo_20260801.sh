set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
source /etc/network_turbo
timeout 15 git ls-remote --heads personal codex/qam-pi0-robotwin
GIT_TERMINAL_PROMPT=0 timeout 60 git push personal HEAD:codex/qam-pi0-robotwin
local_head=$(git rev-parse HEAD)
remote_head=$(timeout 15 git ls-remote --heads personal codex/qam-pi0-robotwin | awk '{print $1}')
printf 'LOCAL_HEAD=%s\nREMOTE_HEAD=%s\n' "$local_head" "$remote_head"
test "$local_head" = "$remote_head"
git rev-list --left-right --count '@{upstream}...HEAD'
git status --short
