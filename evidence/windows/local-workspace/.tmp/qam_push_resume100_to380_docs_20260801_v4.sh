set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
branch=codex/qam-pi0-robotwin
cd "$repo"
local_head=$(git rev-parse HEAD)
test -z "$(git status --short)"

if ! GIT_TERMINAL_PROMPT=0 timeout 60 git push personal "HEAD:$branch"; then
  (
    source /etc/network_turbo >/dev/null 2>&1
    GIT_TERMINAL_PROMPT=0 timeout 60 git push personal "HEAD:$branch"
  )
fi

if ! remote_head=$(timeout 20 git ls-remote personal "refs/heads/$branch" | awk '{print $1}'); then
  remote_head=$(
    source /etc/network_turbo >/dev/null 2>&1
    timeout 20 git ls-remote personal "refs/heads/$branch" | awk '{print $1}'
  )
fi
printf 'local_head=%s\nremote_head=%s\n' "$local_head" "$remote_head"
test "$local_head" = "$remote_head"
git status --short
