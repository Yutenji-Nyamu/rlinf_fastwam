#!/usr/bin/env bash

set +e

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
sudo -S -p '' -l
printf 'sudo_list_rc=%s\n' "$?"
sudo -k

source /etc/profile.d/mihomo-proxy.sh 2>/dev/null
curl -sSIL --max-time 15 --connect-timeout 8 -o /dev/null \
  -w 'modelscope_home http=%{http_code} remote=%{remote_ip} connect=%{time_connect} total=%{time_total} bytes=%{size_download}\n' \
  https://www.modelscope.cn/ 2>&1

exit 0
