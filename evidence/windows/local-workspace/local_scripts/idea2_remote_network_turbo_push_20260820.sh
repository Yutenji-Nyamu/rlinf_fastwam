set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
branch=codex/idea2-dvac-pi0-robotwin
commit=73da63f01d52290c3537f12fb4bfb55cfd82f94c

cd "$target"
test "$(git rev-parse HEAD)" = "$commit"
test "$(git branch --show-current)" = "$branch"
test -z "$(git status --short)"
test -r /etc/network_turbo
test -z "${HTTPS_PROXY:-}${https_proxy:-}${ALL_PROXY:-}${all_proxy:-}"

(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u

  remote_before="$(timeout 15s git ls-remote personal "refs/heads/$branch")"
  printf 'REMOTE_BEFORE=%s\n' "${remote_before:-ABSENT}"
  test -z "$remote_before"
  GIT_TERMINAL_PROMPT=0 timeout 40s \
    git push personal "HEAD:refs/heads/$branch"
  remote_after="$(timeout 15s git ls-remote personal "refs/heads/$branch")"
  printf 'REMOTE_AFTER=%s\n' "$remote_after"
  test "$(printf '%s\n' "$remote_after" | awk '{print $1}')" = "$commit"
)

test -z "${HTTPS_PROXY:-}${https_proxy:-}${ALL_PROXY:-}${all_proxy:-}"
test -z "$(git status --short)"
printf 'PARENT_PROXY_CLEAN=1\n'
printf 'PUSH_VERIFIED=1\n'
