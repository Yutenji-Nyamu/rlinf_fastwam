#!/usr/bin/env bash
set -euo pipefail

TZ=Asia/Shanghai date --iso-8601=seconds
printf 'identity='; id
uptime
awk '/^MemTotal:|^MemAvailable:|^SwapTotal:|^SwapFree:/ {print}' /proc/meminfo
df -hT / /home /data
df -ih / /home /data

echo 'GPU summary:'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo 'GPU processes:'
for process_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu); do
  user=$(ps -o user= -p "$process_pid" | xargs)
  job=none
  if [[ -r "/proc/$process_pid/environ" ]]; then
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ -n "$job" ]] || job=none
  fi
  printf 'pid=%s user=%s job=%s rss_mib=%s cmd=%s\n' "$process_pid" "$user" "$job" \
    "$(ps -o rss= -p "$process_pid" | awk '{printf "%.1f", $1/1024}')" \
    "$(ps -o comm= -p "$process_pid" | xargs)"
done

echo 'per-user RSS/CPU:'
ps -eo user=,rss=,%cpu= | awk '{rss[$1]+=$2; cpu[$1]+=$3} END {for (u in rss) printf "%s rss_gib=%.3f cpu_pct=%.1f\n",u,rss[u]/1024/1024,cpu[u]}' | sort
echo 'largest processes:'
ps -eo user=,pid=,etimes=,%cpu=,%mem=,rss=,comm=,args= --sort=-rss | head -n 22

echo 'network/proxy:'
pgrep -af 'mihomo|clash' || true
ss -ltnp 2>/dev/null | grep -E ':(7890|7891|7892|6389)[[:space:]]' || true
for target in https://github.com https://huggingface.co; do
  code=$(curl -x http://127.0.0.1:7890 -L -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 12 "$target" 2>/dev/null || true)
  printf 'proxy_http target=%s code=%s\n' "$target" "${code:-failed}"
done
