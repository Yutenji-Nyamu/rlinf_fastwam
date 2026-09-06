#!/usr/bin/env bash
set -euo pipefail

date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
uptime

echo '-- gpu --'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '-- gpu owners --'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  [[ -r "/proc/$pid/status" ]] || continue
  ps -o user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= -p "$pid" || true
done

echo '-- memory --'
free -h
printf 'memory_pressure='; tr '\n' ' ' </proc/pressure/memory; echo

echo '-- storage --'
df -hT / /home /data
df -ih / /home /data

echo '-- network --'
printf 'mihomo='; systemctl is-active mihomo 2>/dev/null || true
for spec in \
  'direct_github https://github.com/' \
  'direct_hf https://huggingface.co/api/models?limit=1'; do
  set -- $spec
  code=$(curl --noproxy '*' -L -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 12 "$2" 2>/dev/null || true)
  echo "$1=$code"
done
for spec in \
  'proxy_github https://github.com/' \
  'proxy_hf https://huggingface.co/api/models?limit=1'; do
  set -- $spec
  code=$(curl --proxy http://127.0.0.1:7890 -L -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 12 "$2" 2>/dev/null || true)
  echo "$1=$code"
done

echo '-- user aggregates --'
ps -eo user:20=,rss=,%cpu= | awk '$1 ~ /^(chenyiteng|liwenbo|xiongzizhen|zhangwei|toom|zhuanghuiping|guorenjie|qiufuwen|yanchuhan)$/ {n[$1]++; rss[$1]+=$2; cpu[$1]+=$3} END {for (u in n) printf "%s processes=%d rss_gib=%.3f cpu_sum=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort

echo '-- failed units --'
systemctl --failed --no-pager --no-legend 2>/dev/null || true
echo SZ_SERVER_COMPACT_READONLY_OK
