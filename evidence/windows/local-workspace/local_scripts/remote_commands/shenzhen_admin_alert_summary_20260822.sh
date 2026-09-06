#!/usr/bin/env bash
set -uo pipefail

date --iso-8601=seconds
hostname
id
sudo -S -p '' -v
sudo -n true

printf 'memory_now\n'
free -h
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo

printf 'failed_units\n'
systemctl --failed --no-pager || true

printf 'oom_disk_fs_matches_3d\n'
oom_disk_matches="$(sudo -n journalctl --since '3 days ago' --no-pager -o short-iso 2>/dev/null | grep -Ei 'out of memory|oom-kill|killed process|no space left|disk full|read-only file system|I/O error|ext4-fs error|xfs.*error|nvme.*error' || true)"
printf '%s\n' "$oom_disk_matches" | awk 'NF {count++} END {print count+0}'
printf '%s\n' "$oom_disk_matches" | tail -n 30

printf 'nvidia_xid_matches_3d\n'
xid_matches="$(sudo -n journalctl --since '3 days ago' --no-pager -o short-iso 2>/dev/null | grep -E 'NVRM: Xid' || true)"
printf '%s\n' "$xid_matches" | awk 'NF {count++} END {print count+0}'
printf '%s\n' "$xid_matches" | tail -n 30

printf 'dmesg_oom_disk_fs_matches_since_boot\n'
dmesg_oom_disk="$(sudo -n dmesg --ctime 2>/dev/null | grep -Ei 'out of memory|oom-kill|killed process|no space left|disk full|read-only file system|I/O error|ext4-fs error|xfs.*error|nvme.*error' || true)"
printf '%s\n' "$dmesg_oom_disk" | awk 'NF {count++} END {print count+0}'
printf '%s\n' "$dmesg_oom_disk" | tail -n 30

printf 'dmesg_nvidia_xid_matches_since_boot\n'
dmesg_xid="$(sudo -n dmesg --ctime 2>/dev/null | grep -E 'NVRM: Xid' || true)"
printf '%s\n' "$dmesg_xid" | awk 'NF {count++} END {print count+0}'
printf '%s\n' "$dmesg_xid" | tail -n 30

printf 'current_non_chenyiteng_gpu_processes\n'
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u || true)"
while IFS= read -r pid; do
  [[ -n "$pid" ]] || continue
  ps -o user=,pid=,etimes=,rss=,comm= -p "$pid"
done <<< "$gpu_pids"

printf 'alert_summary_complete\n'
