#!/usr/bin/env bash
set -uo pipefail

date --iso-8601=seconds
hostname
id
sudo -S -p '' -v
sudo -n true

printf 'dmesg_total_lines\n'
sudo -n dmesg 2>/dev/null | wc -l

printf 'dmesg_oom_disk_count\n'
sudo -n dmesg 2>/dev/null | awk 'BEGIN{IGNORECASE=1} /out of memory|oom-kill|killed process|no space left|disk full|read-only file system|I\/O error|ext4-fs error|xfs.*error|nvme.*error/ {count++} END {print count+0}'

printf 'dmesg_xid_count_and_tail\n'
sudo -n dmesg --ctime 2>/dev/null | awk 'index($0,"Xid") {lines[++count]=$0} END {print count+0; start=count-14; if(start<1)start=1; for(i=start;i<=count;i++)print lines[i]}'

printf 'journal_total_lines_3d\n'
sudo -n journalctl --since '3 days ago' --no-pager 2>/dev/null | wc -l

printf 'journal_oom_disk_count\n'
sudo -n journalctl --since '3 days ago' --no-pager 2>/dev/null | awk 'BEGIN{IGNORECASE=1} /out of memory|oom-kill|killed process|no space left|disk full|read-only file system|I\/O error|ext4-fs error|xfs.*error|nvme.*error/ {count++} END {print count+0}'

printf 'kernel_counter_complete\n'
