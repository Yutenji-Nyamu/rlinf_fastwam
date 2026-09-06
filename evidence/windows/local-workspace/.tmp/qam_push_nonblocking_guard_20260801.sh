set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
expected=9e2abc04e8c178575d9b800154d69b9123e73ecb
test "$(git rev-parse HEAD)" = "$expected"
test -z "$(git status --short)"
test -r /etc/network_turbo
(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u
  GIT_TERMINAL_PROMPT=0 timeout 60 git push personal \
    HEAD:refs/heads/codex/qam-pi0-robotwin
  remote_head=$(timeout 15 git ls-remote personal \
    refs/heads/codex/qam-pi0-robotwin | awk '{print $1}')
  printf 'remote_head=%s\n' "$remote_head"
  test "$remote_head" = "$expected"
)
