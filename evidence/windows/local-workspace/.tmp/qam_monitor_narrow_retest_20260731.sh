#!/usr/bin/env bash
set -euo pipefail

monitor=/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh
output=/root/autodl-tmp/qam_resource_monitor_retest_20260731.csv

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
hostname
pwd
id -u
sha256sum "$monitor"
stat -c 'mode=%a size=%s path=%n' "$monitor"

sleep 3 &
probe_pid=$!
bash "$monitor" "$probe_pid" "$output" 1
monitor_status=$?

echo "MONITOR_STATUS=$monitor_status"
echo "CSV_LINES=$(wc -l < "$output")"
head -n 2 "$output"
tail -n 1 "$output"
rm -f -- "$output"
