#!/usr/bin/env bash
set -u

SINCE='2026-08-20 00:00:00'
USERS='chenyiteng liwenbo xiongzizhen zhangwei toom'

printf 'IDENTITY_AND_TIME\n'
hostname
date --iso-8601=seconds
id
uptime
printf 'sudo_noninteractive='; if sudo -n true 2>/dev/null; then echo yes; else echo no; fi

printf 'OS_CPU_MEMORY\n'
cat /etc/os-release | grep -E '^(PRETTY_NAME|VERSION_ID)=' || true
uname -a
lscpu | grep -E '^(CPU\(s\)|On-line CPU|Model name|Socket|Core|Thread|NUMA node)' || true
free -h
printf 'mem_available_kib='; awk '/^MemAvailable:/ {print $2}' /proc/meminfo
printf 'swap_devices\n'; swapon --show --bytes || true
printf 'vmstat_samples\n'; vmstat 1 3

printf 'FILESYSTEMS\n'
findmnt -rno TARGET,SOURCE,FSTYPE,OPTIONS / /home /data 2>/dev/null || true
df -hT / /home /data
df -ih / /home /data
lsblk -o NAME,SIZE,FSTYPE,FSAVAIL,FSUSE%,MOUNTPOINTS,MODEL

printf 'TOP_LEVEL_LAYOUT\n'
for root in /home /data; do
  printf 'root=%s\n' "$root"
  find "$root" -mindepth 1 -maxdepth 1 -printf '%M\t%u:%g\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' 2>/dev/null | sort
done

printf 'RECENT_USER_DIRECTORY_ACTIVITY_METADATA\n'
for user in $USERS; do
  printf 'user=%s\n' "$user"
  for root in "/home/$user" "/data/$user"; do
    if test -d "$root"; then
      printf 'root=%s\n' "$root"
      find "$root" -xdev -mindepth 1 -maxdepth 2 -newermt "$SINCE" \
        -printf '%TY-%Tm-%Td %TH:%TM:%TS\t%u:%g\t%y\t%p\n' 2>/dev/null | sort -r | head -n 80
    fi
  done
done

printf 'LOGINS_NOW\n'
who -a || true
w -h || true
printf 'RECENT_LOGINS\n'
last -F -n 60 2>/dev/null || true
printf 'LASTLOG_RELEVANT\n'
for user in $USERS; do lastlog -u "$user" 2>/dev/null || true; done

printf 'RECENT_AUTH_ACCEPT_AND_SUDO\n'
if test -r /var/log/auth.log; then
  grep -aE 'Accepted (password|publickey)|sudo:.*(COMMAND=|session opened|session closed)' /var/log/auth.log | tail -n 160 || true
else
  printf 'auth_log_not_readable\n'
fi

printf 'ACTIVE_PROCESSES_BY_USER\n'
ps -eo user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= --sort=-rss | head -n 100
printf 'PROCESS_SUMMARY_BY_USER\n'
ps -eo user=,rss=,%cpu= | awk '{n[$1]++; rss[$1]+=$2; cpu[$1]+=$3} END {for (u in n) printf "%s\tprocesses=%d\trss_gib=%.3f\tcpu_sum=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort

printf 'GPU_STATE\n'
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
printf 'GPU_COMPUTE_APPS\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
printf 'GPU_PID_OWNERS\n'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u); do
  test -r "/proc/$pid/status" || continue
  ps -o user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,args= -p "$pid"
done

printf 'SYSTEM_HEALTH\n'
systemctl --failed --no-pager 2>&1 || true
printf 'mihomo_active='; systemctl is-active mihomo 2>/dev/null || true
printf 'docker_active='; systemctl is-active docker 2>/dev/null || true
printf 'recent_kernel_critical\n'
journalctl -k --since "$SINCE" --no-pager 2>/dev/null | \
  grep -aEi 'oom|out of memory|killed process|NVRM|Xid|I/O error|nvme.*(error|critical)|EXT4-fs error|segfault|hardware error' | tail -n 120 || true
printf 'recent_system_error_count='; journalctl --since "$SINCE" -p err..alert --no-pager 2>/dev/null | wc -l
journalctl --since "$SINCE" -p err..alert --no-pager 2>/dev/null | tail -n 100 || true

printf 'NETWORK_LISTEN_SUMMARY\n'
ss -lnt | head -n 80 || true
printf 'proxy_port_7890='; ss -lnt | awk '$4 ~ /127\.0\.0\.1:7890$/ {n++} END {print n+0}'

printf 'SZ_ADMIN_WHOLE_SERVER_AUDIT_OK\n'
