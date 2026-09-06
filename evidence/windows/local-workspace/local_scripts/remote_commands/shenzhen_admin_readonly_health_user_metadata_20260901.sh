#!/usr/bin/env bash
set -u

sudo -S -p '' bash -c '
set -u
export LC_ALL=C

echo "=== identity_time ==="
date -Is
hostname
uptime

echo "=== gpu_state ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true
echo "--- gpu_process_owners ---"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d " " | sort -nu); do
  ps -o user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,comm=,args= -p "$pid" 2>/dev/null \
    | sed -E "s#(token|password|passwd|secret|api[_-]?key)(=|[[:space:]]+)[^[:space:]]+#\1\2<redacted>#Ig" \
    | cut -c1-320
done

echo "=== memory_pressure_storage ==="
free -h
echo "--- swap ---"
swapon --show --bytes 2>/dev/null || true
echo "--- psi_memory ---"
cat /proc/pressure/memory 2>/dev/null || true
echo "--- psi_cpu ---"
cat /proc/pressure/cpu 2>/dev/null || true
echo "--- psi_io ---"
cat /proc/pressure/io 2>/dev/null || true
df -hT / /home /data
df -ih / /home /data

echo "=== ray_and_training_processes ==="
ps -eo user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,stat=,comm=,args= --sort=-rss \
  | grep -E "(raylet|gcs_server|dashboard|RLinf|ray::|train.py|torchrun|accelerate|lerobot|robotwin|FastWAM|python)" \
  | sed -E "s#(token|password|passwd|secret|api[_-]?key)(=|[[:space:]]+)[^[:space:]]+#\1\2<redacted>#Ig" \
  | head -n 100 | cut -c1-360 || true
echo "--- process_summary_by_user ---"
ps -eo user=,rss=,%cpu= | awk "{n[\$1]++; rss[\$1]+=\$2; cpu[\$1]+=\$3} END {for (u in n) printf \"%s processes=%d rss_gib=%.2f cpu_sum=%.1f\\n\",u,n[u],rss[u]/1048576,cpu[u]}" | sort

echo "=== network_basic ==="
ip -br addr 2>/dev/null || true
ip route show default 2>/dev/null || true
ss -s 2>/dev/null || true
echo "--- interface_error_counters ---"
for iface in /sys/class/net/*; do
  n=$(basename "$iface")
  rx=$(cat "$iface/statistics/rx_errors" 2>/dev/null || echo NA)
  tx=$(cat "$iface/statistics/tx_errors" 2>/dev/null || echo NA)
  drop=$(cat "$iface/statistics/rx_dropped" 2>/dev/null || echo NA)
  echo "$n rx_errors=$rx tx_errors=$tx rx_dropped=$drop"
done
echo "--- external_https_probe ---"
for url in https://github.com https://huggingface.co; do
  curl -L -sS -o /dev/null --connect-timeout 5 --max-time 12 -w "$url code=%{http_code} connect=%{time_connect}s total=%{time_total}s remote=%{remote_ip}\\n" "$url" || echo "$url probe_failed"
done

echo "=== users_and_top_level_layout ==="
getent passwd | awk -F: "\$3 >= 1000 && \$3 < 65534 {print \$1,\$3,\$4,\$6,\$7}"
for root in /home /data; do
  echo "--- $root ---"
  find "$root" -mindepth 1 -maxdepth 1 -printf "%M %u:%g %TY-%Tm-%TdT%TH:%TM %y %p -> %l\\n" 2>/dev/null | sort
done

echo "=== recent_7d_top_metadata ==="
for root in /home/* /data/*; do
  test -d "$root" || continue
  owner=$(stat -c "%U:%G" "$root" 2>/dev/null || echo unknown)
  echo "--- root=$root owner=$owner ---"
  find "$root" -xdev -mindepth 1 -maxdepth 3 -newermt "7 days ago" \
    -printf "%TY-%Tm-%TdT%TH:%TM %u:%g %y %s %p -> %l\\n" 2>/dev/null \
    | sort -r | head -n 45
done

echo "=== recent_7d_large_files_metadata ==="
find /home /data -xdev -mindepth 2 -maxdepth 6 -type f -newermt "7 days ago" -size +1G \
  -printf "%TY-%Tm-%TdT%TH:%TM %u:%g %s %p\\n" 2>/dev/null | sort -r | head -n 120 || true

echo "=== git_repo_metadata ==="
find /home /data -xdev -mindepth 2 -maxdepth 6 -name .git \( -type d -o -type f \) -print 2>/dev/null \
  | sort | head -n 100 | while IFS= read -r marker; do
      repo=$(dirname "$marker")
      remote=$(git -c safe.directory="$repo" -C "$repo" remote get-url origin 2>/dev/null || echo no-origin)
      remote=$(printf "%s" "$remote" | sed -E "s#(https?://)[^/@:]+:[^/@]+@#\1<redacted>@#")
      branch=$(git -c safe.directory="$repo" -C "$repo" symbolic-ref --short -q HEAD 2>/dev/null || echo DETACHED)
      head=$(git -c safe.directory="$repo" -C "$repo" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
      ctime=$(git -c safe.directory="$repo" -C "$repo" show -s --format=%cI HEAD 2>/dev/null || echo unknown)
      owner=$(stat -c "%U:%G" "$repo" 2>/dev/null || echo unknown)
      echo "$owner repo=$repo branch=$branch head=$head commit_time=$ctime origin=$remote"
    done

echo "=== recent_hf_model_roots ==="
find /home /data -xdev -mindepth 2 -maxdepth 7 -type d -name "models--*" -newermt "7 days ago" \
  -printf "%TY-%Tm-%TdT%TH:%TM %u:%g %p\\n" 2>/dev/null | sort -r | head -n 100 || true

echo "=== light_security_indicators ==="
echo "--- recent_setuid_or_setgid_user_files ---"
find /home /data -xdev -mindepth 2 -maxdepth 6 -type f -newermt "7 days ago" -perm /6000 \
  -printf "%M %u:%g %TY-%Tm-%TdT%TH:%TM %p\\n" 2>/dev/null | head -n 80 || true
echo "--- suspicious_process_name_patterns ---"
ps -eo user=,pid=,comm=,args= \
  | grep -Ei "xmrig|kinsing|minerd|cryptonight|stratum|kdevtmpfsi|watchbog" \
  | grep -v -E "grep|shenzhen_admin_readonly" | head -n 40 || true
echo "--- failed_services ---"
systemctl --failed --no-pager 2>/dev/null || true
echo "--- recent_kernel_critical_24h ---"
journalctl -k --since "24 hours ago" --no-pager 2>/dev/null \
  | grep -Ei "oom|out of memory|killed process|NVRM|Xid|I/O error|nvme.*(error|critical)|EXT4-fs error|hardware error" \
  | tail -n 80 || true

echo "SZ_ADMIN_READONLY_HEALTH_USER_METADATA_OK"
'
