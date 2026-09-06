set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
expected=24cbc8d20d19161c46da9940b5731127530e911d
cd "$repo"
test "$(git rev-parse HEAD)" = "$expected"
test -z "$(git status --short)"
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
