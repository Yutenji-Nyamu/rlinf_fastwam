#!/usr/bin/env bash

set +e

section() {
  printf '\n===== %s =====\n' "$1"
}

section timestamp_identity
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
hostname
id
groups
sudo -S -p '' true
printf 'sudo_auth_rc=%s\n' "$?"
sudo -n -l 2>&1

section accounts_sessions_and_activity
sudo -n getent group sudo adm lxd labdata docker
who -a
w -h
last -F -n 20 2>&1
printf '%s\n' '-- active users by process count/rss/cpu --'
sudo -n ps -eo user=,rss=,%cpu=,comm= | awk '{n[$1]++; rss[$1]+=$2; cpu[$1]+=$3} END {for (u in n) printf "%s processes=%d rss_kib=%d cpu_sum=%.1f\n",u,n[u],rss[u],cpu[u]}' | sort -k3,3nr
printf '%s\n' '-- top RSS, command name only --'
sudo -n ps -eo user,pid,ppid,stat,%cpu,%mem,rss,vsz,etimes,comm --sort=-rss | head -n 45
printf '%s\n' '-- top CPU, command name only --'
sudo -n ps -eo user,pid,ppid,stat,%cpu,%mem,rss,etimes,comm --sort=-%cpu | head -n 35

section gpu_process_ownership
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  test -d "/proc/$pid" || continue
  owner=$(sudo -n stat -c '%U' "/proc/$pid" 2>/dev/null)
  name=$(sudo -n awk '/^Name:/{print $2}' "/proc/$pid/status" 2>/dev/null)
  rss=$(sudo -n awk '/^VmRSS:/{print $2" "$3}' "/proc/$pid/status" 2>/dev/null)
  printf 'pid=%s owner=%s name=%s rss=%s\n' "$pid" "$owner" "$name" "$rss"
done

section services_and_health
sudo -n systemctl --failed --no-pager
for unit in ssh.service mihomo.service docker.service containerd.service; do
  printf -- '--- %s ---\n' "$unit"
  sudo -n systemctl show "$unit" -p LoadState -p ActiveState -p SubState -p UnitFileState -p MainPID -p ExecMainStatus --no-pager 2>&1
done
sudo -n ss -lntp | grep -E ':(22|7890|9090)\b' || true
printf '%s\n' '-- kernel warning/error since boot, last 80 --'
sudo -n journalctl -k -p warning..alert -b --no-pager -o short-iso 2>&1 | tail -n 80
printf '%s\n' '-- service error since midnight, last 80 --'
sudo -n journalctl -p err..alert --since 'today 00:00' --no-pager -o short-iso 2>&1 | tail -n 80
printf '%s\n' '-- auth summary, last 5000 lines --'
if sudo -n test -r /var/log/auth.log; then
  sudo -n tail -n 5000 /var/log/auth.log | awk '
    /Accepted (password|publickey)/ {accepted++; for (i=1; i<=NF; i++) if ($i=="for" && (i+1)<=NF) user[$(i+1)]++}
    /Failed password/ {failed++}
    /Invalid user/ {invalid++}
    END {printf "accepted=%d failed_password=%d invalid_user=%d\n",accepted,failed,invalid; for (u in user) printf "accepted_user=%s count=%d\n",u,user[u]}'
else
  printf 'auth_log_unreadable_or_absent\n'
fi

section storage_filesystem_health
df -hT / /home /data
df -ih / /home /data
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,ROTA,MODEL
cat /proc/mdstat 2>&1
sudo -n journalctl -k -b --no-pager -o cat 2>&1 | grep -Ei 'I/O error|EXT4-fs error|XFS.*error|nvme.*(reset|timeout|fatal)|out of memory|oom-kill|killed process' | tail -n 80

section network_and_mihomo_profile_metadata
ip route
resolvectl status 2>&1 | sed -n '1,140p'
sudo -n systemctl cat mihomo.service --no-pager 2>&1 | sed -E 's#(https?://)[^ ]+#URL_REDACTED#g'
sudo -n find /etc/mihomo -maxdepth 2 -type f -printf '%M %U:%G %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>&1 | sed -n '1,120p'
sudo -n grep -RIlE 'proxy-providers:|subscription-userinfo|^[[:space:]]*url:' /etc/mihomo 2>/dev/null | sed -n '1,80p'

section fastwam_and_ppo_process_boundary
sudo -n ps -eo user,pid,ppid,stat,%cpu,%mem,rss,etimes,comm,args | grep -Ei 'ppo-formal100-4gpu128train64eval-official-v1|train_embodied_agent.py|eval_embodied_agent.py|fastwam|pip.*torch|conda.*fastwam' | grep -v -E 'grep -Ei|shenzhen_full_readonly_audit' | sed -E 's#(https?://)[^ ]+#URL_REDACTED#g' | head -n 140

section end
sudo -k
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
exit 0
