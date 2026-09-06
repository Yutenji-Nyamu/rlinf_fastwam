#!/usr/bin/env bash
set -uo pipefail

printf 'audit_identity\n'
date --iso-8601=seconds
hostname
id
uptime
who -a

printf 'filesystem_capacity\n'
df -hT
df -ih

printf 'mounts\n'
findmnt -rn -o TARGET,SOURCE,FSTYPE,OPTIONS | sort

printf 'sudo_authentication\n'
sudo -S -p '' -v
sudo -n true
printf 'sudo_readonly=PASS\n'

printf 'root_filesystem_top_level_bytes\n'
sudo -n du -x -B1 -d1 / 2>/dev/null | sort -n

printf 'home_top_level_bytes\n'
sudo -n du -x -B1 -d1 /home 2>/dev/null | sort -n

printf 'data_top_level_bytes\n'
sudo -n du -x -B1 -d1 /data 2>/dev/null | sort -n

printf 'largest_user_rss_aggregate_kib\n'
ps -eo user=,rss= | awk '{rss[$1]+=$2} END {for (u in rss) printf "%s %d\n",u,rss[u]}' | sort -k2,2nr | head -n 30

printf 'largest_processes_no_arguments\n'
ps -eo user=,pid=,ppid=,etimes=,rss=,stat=,comm= --sort=-rss | head -n 61

printf 'gpu_summary\n'
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,pstate,temperature.gpu --format=csv,noheader
printf 'gpu_compute_processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

printf 'recent_login_sessions\n'
last -F -n 60

printf 'failed_units\n'
systemctl --failed --no-pager || true

printf 'kernel_storage_memory_gpu_alerts_since_boot\n'
sudo -n dmesg --ctime 2>/dev/null | grep -Ei 'out of memory|oom|killed process|no space left|read-only file system|I/O error|ext4-fs error|xfs.*error|nvme.*error|nvrm|xid|gpu.*fallen off|segfault' | tail -n 200 || true

printf 'journal_storage_memory_gpu_alerts_3d\n'
sudo -n journalctl --since '3 days ago' --no-pager -o short-iso 2>/dev/null | grep -Ei 'out of memory|oom-kill|killed process|no space left|disk full|read-only file system|I/O error|ext4-fs error|xfs.*error|nvme.*error|nvrm|xid|gpu.*fallen off' | tail -n 200 || true

printf 'audit_complete\n'
