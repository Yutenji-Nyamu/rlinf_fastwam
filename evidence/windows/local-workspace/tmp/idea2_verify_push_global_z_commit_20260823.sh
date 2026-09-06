set -eu

child=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
branch=refs/heads/codex/idea2-dvac-residual-downweight
expected=afdaa2e2aa59aa16128e89f47eb4aaf7a64badd8
previous=eb2a09176c362c7386895ca4f3680b92aeb0ee5b

echo '=== local ==='
git -C "$child" rev-parse HEAD
git -C "$child" status --short

echo '=== bounded remote verify/push ==='
(
  source /etc/network_turbo >/dev/null 2>&1
  before=$(timeout 15 git -C "$child" ls-remote personal "$branch" | awk '{print $1}')
  echo "remote_before=$before"
  if [ "$before" = "$previous" ]; then
    timeout 40 env GIT_TERMINAL_PROMPT=0 git -C "$child" push personal HEAD:codex/idea2-dvac-residual-downweight
  elif [ "$before" != "$expected" ]; then
    echo "unexpected remote head: $before" >&2
    exit 3
  fi
  after=$(timeout 15 git -C "$child" ls-remote personal "$branch" | awk '{print $1}')
  echo "remote_after=$after"
  test "$after" = "$expected"
)

echo '=== parent proxy state ==='
env | grep -iE '^(http|https|all)_proxy=' || true
