set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
branch=codex/idea2-dvac-pi0-robotwin
old_head=73da63f01d52290c3537f12fb4bfb55cfd82f94c
new_head=61996e15cc7f5a32bd6012b61b20893d94636c82

cd "$target"
test "$(git rev-parse HEAD)" = "$new_head"
test "$(git branch --show-current)" = "$branch"
test -z "$(git status --porcelain)"
test "$(git rev-parse personal/$branch)" = "$old_head"
test -r /etc/network_turbo
test -z "${HTTPS_PROXY-}${https_proxy-}${ALL_PROXY-}${all_proxy-}"

remote_after=$(
  set -euo pipefail
  source /etc/network_turbo >/dev/null 2>&1
  remote_before=$(timeout 15s git ls-remote personal "refs/heads/$branch")
  test "${remote_before%%[[:space:]]*}" = "$old_head"
  GIT_TERMINAL_PROMPT=0 timeout 40s \
    git push personal "HEAD:refs/heads/$branch"
  timeout 15s git ls-remote personal "refs/heads/$branch"
)

test "${remote_after%%[[:space:]]*}" = "$new_head"
test -z "${HTTPS_PROXY-}${https_proxy-}${ALL_PROXY-}${all_proxy-}"
test "$(git rev-parse personal/$branch)" = "$new_head"
printf 'REMOTE_BEFORE=%s\nREMOTE_AFTER=%s\nPARENT_PROXY_CLEAN=1\nFAST_FORWARD_PUSH_VERIFIED=1\n' \
  "$old_head" "$new_head"

