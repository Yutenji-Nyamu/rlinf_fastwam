set -eu

export LC_ALL=C

sudo -S -p '' true

echo '=== IDENTITY_TIME ==='
date -Is
hostname
id
uptime

echo '=== MEMORY_LOAD ==='
free -h
swapon --show --bytes || true
cat /proc/pressure/memory 2>/dev/null || true
vmstat 1 2 | tail -n 2

echo '=== FILESYSTEMS ==='
df -hT / /home /data
df -ih / /home /data

echo '=== GPU_SUMMARY ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits

echo '=== GPU_COMPUTE_PROCESSES ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo '=== GPU_PID_OWNERS ==='
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -nu); do
  sudo -n ps -p "$pid" -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,comm=,args= 2>/dev/null || true
done

echo '=== TOP_USER_PROCESSES ==='
sudo -n ps -eo user:18,pid,ppid,etime,%cpu,%mem,rss,comm,args --sort=-%cpu \
  | awk 'NR==1 || ($1 != "root" && $1 != "messagebus" && $1 != "systemd+" && $1 != "nobody")' \
  | head -n 80

echo '=== USER_PROCESS_SUMMARY ==='
sudo -n ps -eo user=,comm= | awk '
  $1 != "root" && $1 != "messagebus" && $1 != "systemd+" && $1 != "nobody" {
    key=$1 SUBSEP $2; c[key]++
  }
  END { for (k in c) { split(k,a,SUBSEP); print a[1],a[2],c[k] } }
' | sort

echo '=== PROXY_SERVICE ==='
systemctl is-active mihomo 2>/dev/null || true
systemctl show mihomo -p ActiveState -p SubState -p NRestarts -p MainPID --no-pager 2>/dev/null || true
sudo -n ps -C mihomo -o user=,pid=,etime=,%cpu=,%mem=,rss=,comm=,args= 2>/dev/null || true
sudo -n ss -ltnp 2>/dev/null | grep -E ':(7890|7891|9090)\b' || true

echo '=== NETWORK_PROBES ==='
set +u
if [ -r /etc/profile.d/mihomo-proxy.sh ]; then
  . /etc/profile.d/mihomo-proxy.sh
fi
set -u
for target in https://github.com/ https://huggingface.co/; do
  code=$(curl -L -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 12 "$target" 2>&1) && rc=0 || rc=$?
  printf 'proxy/default target=%s rc=%s http=%s\n' "$target" "$rc" "$code"
done
for target in https://github.com/ https://huggingface.co/; do
  code=$(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    curl -L -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 12 "$target" 2>&1) && rc=0 || rc=$?
  printf 'direct target=%s rc=%s http=%s\n' "$target" "$rc" "$code"
done

echo '=== KERNEL_HEALTH_TAIL ==='
sudo -n journalctl -k --since '-6 hours' --no-pager 2>/dev/null \
  | grep -Ei 'oom|out of memory|killed process|nvrm|xid|i/o error|ext4-fs error|xfs.*error' \
  | tail -n 40 || true

echo '=== END ==='
