set -euo pipefail
source_dir=/root/autodl-tmp/oracles/qam-2726d767
failed_dir=/root/autodl-tmp/oracles/qam-2726d767.failed-bundle-20260731
url=https://github.com/ColinQiyangLi/qam.git
commit=2726d767c9a0a7a46d49693f0391f73dc2cf58ac

if test -e "$source_dir"; then
  test -d "$source_dir"
  test "$(readlink -f "$source_dir")" = "$source_dir"
  test ! -e "$failed_dir"
  mv "$source_dir" "$failed_dir"
else
  test ! -e "$failed_dir"
fi

env | grep -iE '^(http|https|all)_proxy=' || true
git config --get http.version || printf 'DEFAULT\n'
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com || true
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'api code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://api.github.com || true
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'raw code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://raw.githubusercontent.com || true
GIT_TERMINAL_PROMPT=0 timeout 15 git ls-remote --heads "$url"

git init "$source_dir"
git -C "$source_dir" remote add origin "$url"
GIT_TERMINAL_PROMPT=0 timeout 120 \
  git -C "$source_dir" fetch --depth=1 origin "$commit"
git -C "$source_dir" checkout --detach FETCH_HEAD
test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$source_dir" status --short)"
git -C "$source_dir" log -1 --format='%H %cI %s'
