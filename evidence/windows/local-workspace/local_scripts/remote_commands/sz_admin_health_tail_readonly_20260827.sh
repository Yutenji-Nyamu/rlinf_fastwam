#!/usr/bin/env bash
set -euo pipefail

TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
echo 'sessions:'
who | awk '{print $1, $2, $3, $4}' | sort -u
echo 'failed_units:'
systemctl --failed --no-legend --plain 2>/dev/null || true
echo 'recent_kernel_faults:'
journalctl -k --since '6 hours ago' --no-pager 2>/dev/null \
  | grep -Ei 'out of memory|oom-kill|NVRM|Xid|I/O error|nvme.*error|EXT4-fs error' \
  | tail -n 12 || true
