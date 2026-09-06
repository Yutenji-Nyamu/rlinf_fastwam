#!/usr/bin/env bash
set -euo pipefail

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' '=== Fast-WAM release target ==='
find /data/chenyiteng/models/fastwam/release-8eaceeb \
  -maxdepth 4 -type f \
  \( -name '*.incomplete' -o -name 'robotwin_uncond_3cam_384.pt' -o -name 'robotwin_uncond_3cam_384_dataset_stats.json' \) \
  -printf 'bytes=%s mtime=%TY-%Tm-%TdT%TH:%TM:%TS%Tz path=%p\n' 2>/dev/null \
  | sort || true
printf '%s\n' '=== owned download process ==='
ps -u chenyiteng -o pid=,etimes=,rss=,stat=,args= \
  | grep -E 'huggingface-cli download yuanty/fastwam|shenzhen_fastwam_fw_sz_300' \
  | grep -v grep || true
printf '%s\n' '=== capacity ==='
df -B1 --output=target,avail /data /home
