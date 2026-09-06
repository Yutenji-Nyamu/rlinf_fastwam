set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
cd "$repo"
test -r /etc/network_turbo
test "$(git rev-parse HEAD)" = dc3711950a2cad8222ba72bfe0c6de2f7a5babdb
test -z "$(git status --short)"
(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u
  before=$(timeout 15 git ls-remote personal \
    refs/heads/codex/qam-pi0-robotwin | awk '{print $1}')
  test "$before" = d5f6d7d1da0fc355a71ca653be027282cad040d2
  GIT_TERMINAL_PROMPT=0 timeout 60 git push personal \
    HEAD:refs/heads/codex/qam-pi0-robotwin
  after=$(timeout 15 git ls-remote personal \
    refs/heads/codex/qam-pi0-robotwin | awk '{print $1}')
  printf 'REMOTE_AFTER=%s\n' "$after"
  test "$after" = "$(git rev-parse HEAD)"
)
test -z "$(git status --short)"
