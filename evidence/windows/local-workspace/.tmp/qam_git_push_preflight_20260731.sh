set -u

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
branch=codex/qam-pi0-robotwin

date '+TIME=%F %T %Z'
printf 'IDENTITY=%s %s %s\n' "$(hostname)" "$PWD" "$(id -u)"
printf 'BRANCH=%s\n' "$(git -C "$repo" branch --show-current)"
printf 'HEAD=%s\n' "$(git -C "$repo" rev-parse HEAD)"
printf 'TREE=%s\n' "$(git -C "$repo" rev-parse 'HEAD^{tree}')"
printf 'STATUS_BEGIN\n'
git -C "$repo" status --short
printf 'STATUS_END\n'
printf 'REMOTES='
git -C "$repo" remote
printf 'UPSTREAM='
git -C "$repo" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' \
  2>/dev/null || printf 'NONE\n'

env | grep -iE '^(http|https|all)_proxy=' || true
git -C "$repo" config --get http.version || printf 'HTTP_VERSION=DEFAULT\n'
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'api code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://api.github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'raw code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://raw.githubusercontent.com
timeout 15 git -C "$repo" ls-remote --heads personal "$branch"
