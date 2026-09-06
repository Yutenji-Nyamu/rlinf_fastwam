#!/usr/bin/env bash
set -euo pipefail

file=/root/autodl-tmp/RoboTwin_RLinf/envs/_base_task.py
echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
hostname
pwd
id -u
for range in 1735,1810 1945,1990 2295,2340 2440,2490; do
  start=${range%,*}
  end=${range#*,}
  echo "LINES_${start}_${end}"
  sed -n "${start},${end}p" "$file"
done
