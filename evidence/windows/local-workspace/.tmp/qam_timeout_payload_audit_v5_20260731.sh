#!/usr/bin/env bash
set -euo pipefail

file=/root/autodl-tmp/RLinf_qam_pi0_robotwin/rlinf/envs/robotwin/robotwin_env.py
echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
hostname
pwd
id -u
sed -n '1,120p' "$file"
sed -n '280,380p' "$file"
