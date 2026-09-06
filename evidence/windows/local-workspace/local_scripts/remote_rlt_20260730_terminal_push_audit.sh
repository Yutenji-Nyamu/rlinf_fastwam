#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
cd "${repo}"

printf 'audit_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'branch\t%s\n' "$(git branch --show-current)"
printf 'head\t%s\n' "$(git rev-parse HEAD)"
printf 'dirty_count\t%s\n' "$(git status --short | wc -l)"
read -r left right < <(git rev-list --left-right --count '@{upstream}...HEAD')
printf 'left_right\t%s/%s\n' "${left}" "${right}"
printf 'push_processes_begin\n'
pgrep -af 'git push|git-remote-https|remote-https' || true
printf 'push_processes_end\n'

for item in \
  'main https://github.com' \
  'api https://api.github.com' \
  'raw https://raw.githubusercontent.com'; do
  read -r name url <<<"${item}"
  result="$(
    curl -I -L -sS -o /dev/null \
      --connect-timeout 7 --max-time 10 \
      -w 'code=%{http_code} connect=%{time_connect} total=%{time_total}' \
      "${url}" 2>&1
  )" || true
  printf '%s\t%s\n' "${name}" "${result//$'\n'/ }"
done

printf 'ls_remote_begin\n'
timeout 15 git ls-remote \
  personal refs/heads/codex/rlt-pi0-robotwin \
  || true
printf 'ls_remote_end\n'
