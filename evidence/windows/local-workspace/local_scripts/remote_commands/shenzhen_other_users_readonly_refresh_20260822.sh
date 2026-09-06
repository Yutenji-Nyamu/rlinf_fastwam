#!/usr/bin/env bash
set -uo pipefail

section() { printf '\n===== %s =====\n' "$1"; }

section identity_and_sessions
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
hostname
id
uptime
sudo -S -p '' true
printf 'sudo_auth_rc=%s\n' "$?"
who -a
w -h
last -F -n 12 2>&1

section per_user_resources
sudo -n ps -eo user=,rss=,%cpu=,comm= | awk '{n[$1]++; rss[$1]+=$2; cpu[$1]+=$3} END {for (u in n) printf "%s processes=%d rss_kib=%d cpu_sum=%.1f\n",u,n[u],rss[u],cpu[u]}' | sort -k3,3nr
printf '%s\n' '-- top non-chenyiteng user processes; command name only --'
sudo -n ps -eo user,pid,ppid,stat,%cpu,%mem,rss,etimes,comm --sort=-rss | awk 'NR==1 || ($1!="chenyiteng" && $1!="root")' | head -n 50
printf '%s\n' '-- top root processes; command name only --'
sudo -n ps -eo user,pid,ppid,stat,%cpu,%mem,rss,etimes,comm --sort=-rss | awk 'NR==1 || $1=="root"' | head -n 25

section gpu_process_owners
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  test -d "/proc/$pid" || continue
  printf 'pid=%s owner=%s comm=%s rss_kib=%s\n' "$pid" \
    "$(sudo -n stat -c '%U' "/proc/$pid" 2>/dev/null)" \
    "$(sudo -n cat "/proc/$pid/comm" 2>/dev/null)" \
    "$(sudo -n awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)"
done

section system_storage_health
sudo -n systemctl --failed --no-pager
for unit in ssh.service mihomo.service docker.service containerd.service; do
  sudo -n systemctl show "$unit" -p ActiveState -p SubState -p MainPID -p ExecMainStatus --no-pager 2>&1
done
df -hT / /home /data
df -ih / /home /data
printf '%s\n' '-- severe kernel matches since midnight --'
sudo -n journalctl -k --since 'today 00:00' --no-pager -o short-iso 2>&1 | grep -Ei 'NVRM.*Xid|I/O error|EXT4-fs error|XFS.*error|nvme.*(reset|timeout|fatal)|out of memory|oom-kill|killed process' | tail -n 40 || true

section end
sudo -k
TZ=Asia/Shanghai date --iso-8601=seconds
exit 0
