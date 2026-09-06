#!/usr/bin/env bash
set -u
id
hostname
sudo -S -p '' /bin/bash -c '
date -Is
id
printf "KERNEL_INCIDENT_WINDOW\n"
journalctl -k --since "2026-09-03 14:45:00 UTC" --until "2026-09-03 16:10:00 UTC" -p warning --no-pager -n 100
printf "KERNEL_LAST24H_HARDWARE_MEMORY\n"
journalctl -k --since "24 hours ago" --no-pager | grep -Ei "out of memory|oom-kill|killed process|NVRM.*Xid|nvme.*(error|reset|timeout)|I/O error|EXT4-fs error|MCE|EDAC.*error" | tail -n 50
printf "SERVICES\n"
systemctl is-active ssh mihomo
systemctl --failed --no-pager --no-legend
printf "DISK_INVENTORY\n"
lsblk -d -o NAME,TYPE,SIZE,MODEL
printf "NVME_SMART\n"
if command -v smartctl >/dev/null; then
  smartctl -H -A /dev/nvme0n1
  smartctl -H -A /dev/nvme1n1
elif command -v nvme >/dev/null; then
  nvme smart-log /dev/nvme0
  nvme smart-log /dev/nvme1
else printf "No existing SMART utility; nothing installed.\n"; fi
'
