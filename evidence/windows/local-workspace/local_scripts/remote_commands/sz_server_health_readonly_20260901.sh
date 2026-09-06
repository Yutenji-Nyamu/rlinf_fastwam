#!/usr/bin/env bash
set -u

echo '=== TIME_ID_LOAD ==='
TZ=Asia/Shanghai date --iso-8601=seconds
id
uptime
printf 'loadavg '; cat /proc/loadavg

echo '=== MEMORY_PRESSURE ==='
awk '/^MemTotal:|^MemAvailable:|^Buffers:|^Cached:|^SwapTotal:|^SwapFree:/ {print}' /proc/meminfo
printf 'memory_psi '; tr '\n' ' ' </proc/pressure/memory; echo
printf 'cpu_psi '; tr '\n' ' ' </proc/pressure/cpu; echo
printf 'io_psi '; tr '\n' ' ' </proc/pressure/io; echo

echo '=== FILESYSTEMS ==='
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS / /home /data 2>/dev/null || true
df -hT / /home /data
df -ih / /home /data

echo '=== GPU_SUMMARY ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '=== GPU_COMPUTE_PROCESSES ==='
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | sort -nu)
if [[ -z "${gpu_pids}" ]]; then
  echo none
else
  for process_pid in ${gpu_pids}; do
    ps -o user=,pid=,ppid=,etimes=,%cpu=,%mem=,rss=,comm=,args= -p "${process_pid}" | sed -e 's/^[[:space:]]*//'
  done
fi

echo '=== USER_RESOURCE_SUMMARY ==='
ps -eo user=,rss=,%cpu= | awk '{rss[$1]+=$2; cpu[$1]+=$3; n[$1]++} END {for (u in rss) printf "%s processes=%d rss_gib=%.2f cpu_pct=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort -k3,3nr
echo '=== TOP_MEMORY_PROCESSES ==='
ps -eo user=,pid=,etimes=,%cpu=,%mem=,rss=,comm=,args= --sort=-rss | head -n 25
echo '=== TOP_CPU_PROCESSES ==='
ps -eo user=,pid=,etimes=,%cpu=,%mem=,rss=,comm=,args= --sort=-%cpu | head -n 20

echo '=== KEY_SERVICES ==='
pgrep -af 'raylet|gcs_server|mihomo|clash|torchrun|run_embodiment|main_ppo|main_grpo' || true
systemctl is-active mihomo 2>/dev/null || true
systemctl --no-pager --failed 2>/dev/null | head -n 20 || true

echo '=== NETWORK ==='
ip -brief link 2>/dev/null || true
ip route show default 2>/dev/null || true
ss -s 2>/dev/null || true
for iface in /sys/class/net/*; do
  name=$(basename "${iface}")
  [[ "${name}" == lo ]] && continue
  state=$(cat "${iface}/operstate" 2>/dev/null || true)
  speed=$(cat "${iface}/speed" 2>/dev/null || true)
  printf 'iface=%s state=%s speed_mbps=%s\n' "${name}" "${state:-unknown}" "${speed:-unknown}"
done
for mode in direct proxy; do
  for target in https://github.com/robots.txt https://huggingface.co/robots.txt; do
    if [[ "${mode}" == proxy ]]; then
      extra=(-x http://127.0.0.1:7890)
    else
      extra=()
    fi
    result=$(curl "${extra[@]}" -L -sS -o /dev/null --connect-timeout 5 --max-time 15 -w 'code=%{http_code} connect=%{time_connect} total=%{time_total} bytes=%{size_download} speed_Bps=%{speed_download}' "${target}" 2>/dev/null || true)
    printf '%s %s %s\n' "${mode}" "${target}" "${result:-failed}"
  done
done

echo '=== LOGGED_IN_USERS ==='
who || true
