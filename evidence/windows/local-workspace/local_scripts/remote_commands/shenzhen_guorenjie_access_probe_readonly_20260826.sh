#!/usr/bin/env bash
set -u

date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
hostname
id
printf 'sudo_noninteractive='; sudo -n -u guorenjie true >/dev/null 2>&1 && echo yes || echo no
printf 'home_metadata='; stat -c '%A %U:%G %s %n' /home/guorenjie 2>&1 || true
printf 'home_list='; ls -ld /home/guorenjie 2>&1 || true

echo '-- gpu processes --'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  ps -o user:16=,pid=,ppid=,lstart=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= -p "$pid" 2>/dev/null || true
done

echo SZ_GUORENJIE_ACCESS_PROBE_OK
