set -eu

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
branch=codex/qam-pi0-robotwin
expected_head=7bc5f87086035087adf6d44ddda76eb5a9e54ee8
expected_tree=6dc9124ba63b5712918ba2dbdcffde203cfb5eed

test "$(git -C "$repo" branch --show-current)" = "$branch"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$repo" rev-parse 'HEAD^{tree}')" = "$expected_tree"
test -z "$(git -C "$repo" status --short)"
test -r /etc/network_turbo

printf 'PARENT_PROXY_BEFORE='
env | grep -iE '^(http|https|all)_proxy=' || printf 'NONE\n'

(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u

  printf 'TURBO_PROXY_STATE='
  if test -n "${http_proxy:-}${https_proxy:-}"; then
    printf 'SET\n'
  else
    printf 'MISSING\n' >&2
    exit 1
  fi

  remote_before=$(
    timeout 15 git -C "$repo" ls-remote personal \
      "refs/heads/$branch" | awk 'NR == 1 {print $1}'
  )
  if test -n "$remote_before"; then
    printf 'REMOTE_BEFORE=%s\n' "$remote_before"
    if git -C "$repo" merge-base --is-ancestor "$remote_before" "$expected_head"; then
      :
    else
      printf 'REMOTE_NOT_ANCESTOR=%s\n' "$remote_before" >&2
      exit 1
    fi
  else
    printf 'REMOTE_BEFORE=ABSENT\n'
  fi

  GIT_TERMINAL_PROMPT=0 timeout 60 \
    git -C "$repo" push --set-upstream personal \
    "HEAD:refs/heads/$branch"

  remote_after=$(
    timeout 15 git -C "$repo" ls-remote personal \
      "refs/heads/$branch" | awk 'NR == 1 {print $1}'
  )
  test "$remote_after" = "$expected_head"
  printf 'REMOTE_AFTER=%s\n' "$remote_after"
)

printf 'PARENT_PROXY_AFTER='
env | grep -iE '^(http|https|all)_proxy=' || printf 'NONE\n'
test "$(git -C "$repo" rev-parse '@{upstream}')" = "$expected_head"
printf 'AHEAD_BEHIND='
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
test -z "$(git -C "$repo" status --short)"
printf 'QAM_TURBO_PUSH_OK=%s\n' "$expected_head"
