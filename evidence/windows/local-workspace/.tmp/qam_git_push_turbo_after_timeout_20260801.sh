set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test "$(git rev-parse HEAD)" = dc3711950a2cad8222ba72bfe0c6de2f7a5babdb
test -z "$(git status --short)"
test -r /etc/network_turbo
(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u
  GIT_TERMINAL_PROMPT=0 timeout 60 git push personal \
    HEAD:refs/heads/codex/qam-pi0-robotwin
  after=$(timeout 15 git ls-remote personal \
    refs/heads/codex/qam-pi0-robotwin | awk '{print $1}')
  printf 'REMOTE_AFTER=%s\n' "$after"
  test "$after" = "$(git rev-parse HEAD)"
)
