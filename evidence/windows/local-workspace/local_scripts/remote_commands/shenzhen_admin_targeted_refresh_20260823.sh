#!/usr/bin/env bash
set -u

printf '=== SNAPSHOT ===\n'
date --iso-8601=seconds
uptime
free -h
swapon --show --bytes || true
vmstat 1 3
df -hT / /home /data
df -ih / /home /data

printf '=== GPU ===\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
printf '%s\n' '--- GPU process owners ---'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u); do
  ps -o user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,cmd= -p "$pid" 2>/dev/null || true
done

printf '=== USER ACTIVITY BOUNDARY ===\n'
who -u || true
printf '%s\n' '--- current non-system user processes ---'
ps -eo user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,cmd= --sort=-rss \
  | awk '$1 ~ /^(chenyiteng|liwenbo|toom|xiongzizhen|zhangwei)$/ {print}' \
  | head -n 60 || true
printf '%s\n' '--- recent login sessions ---'
last -F -n 40 chenyiteng liwenbo toom xiongzizhen zhangwei 2>/dev/null || true
printf '%s\n' '--- top-level directory mtimes only ---'
for root in /home /data; do
  find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%TY-%Tm-%TdT%TH:%TM:%TS %u %g %m %p\n' 2>/dev/null | sort
done

printf '=== SERVICES ===\n'
systemctl is-active ssh mihomo docker containerd 2>/dev/null || true
systemctl --failed --no-pager --plain 2>/dev/null || true
ss -lntp 2>/dev/null | awk '$4 ~ /:22$|:7890$|:8265$/ {print}' || true

printf '=== TARGETED SYSTEM ANOMALIES SINCE AUG20 ===\n'
for pattern in 'Out of memory|oom-kill|Killed process' 'NVRM: Xid' 'segfault' 'I/O error|EXT4-fs error|nvme.*(reset|timeout|abort)' ; do
  printf 'pattern=%s count=' "$pattern"
  journalctl --since '2026-08-20 00:00:00' --no-pager -o cat 2>/dev/null | grep -Eic "$pattern" || true
done
printf '%s\n' '--- matching kernel/system lines ---'
journalctl --since '2026-08-20 00:00:00' --no-pager -o short-iso 2>/dev/null \
  | grep -Ei 'Out of memory|oom-kill|Killed process|NVRM: Xid|segfault|I/O error|EXT4-fs error|nvme.*(reset|timeout|abort)' \
  | tail -n 60 || true

printf '=== AUTH SUMMARY SINCE AUG20 ===\n'
if test -r /var/log/auth.log; then
  printf 'accepted_by_user\n'
  grep -hE '^Aug (20|21|22|23).*sshd.*Accepted ' /var/log/auth.log 2>/dev/null \
    | sed -nE 's/.* for ([^ ]+) from ([^ ]+).*/\1 \2/p' | sort | uniq -c | sort -nr || true
  printf 'failed_password_count='
  grep -hEic '^Aug (20|21|22|23).*sshd.*Failed password' /var/log/auth.log 2>/dev/null || true
  printf 'invalid_user_count='
  grep -hEic '^Aug (20|21|22|23).*sshd.*Invalid user' /var/log/auth.log 2>/dev/null || true
  printf 'kex_or_banner_noise_count='
  grep -hEic '^Aug (20|21|22|23).*sshd.*(kex_exchange_identification|banner exchange)' /var/log/auth.log 2>/dev/null || true
fi

printf 'SZ_ADMIN_TARGETED_REFRESH_OK\n'
