#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
ACTION="$ROOT/dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"
ST="$ROOT/dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1"
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

printf 'NOW=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
printf 'UPTIME='; uptime

for item in "ACTION_FIX:$ACTION" "ST_HALF:$ST"; do
  label=${item%%:*}; run=${item#*:}
  printf '\n=== %s ===\n' "$label"
  pid=$(cat "$run/runtime/wrapper.pid" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then alive=1; else alive=0; fi
  printf 'wrapper_pid=%s alive=%s exit=' "$pid" "$alive"
  cat "$run/runtime/exit_code.txt" 2>/dev/null || echo pending
  printf 'fatal_count='
  grep -aEic 'Traceback|CUDA out of memory|Error executing job|RayActorError|WorkerCrashedError|NCCL.*(error|failed)|non.?finite' "$run/runtime/driver.log" 2>/dev/null || true
  printf 'latest_complete='
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$run/runtime/driver.log" 2>/dev/null | tail -n1 || echo none
  printf 'latest_progress='
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100|Rollout Epoch:[[:space:]]+[0-9]+/4|Generating Rollout Epochs:[[:space:]]+[0-9]+/4' "$run/runtime/driver.log" 2>/dev/null | tail -n3 | tr '\n' ';'; echo
  grep -aE 'train/.*success|actor/grad_norm|actor/dvac_weight_(mean|ess_fraction)|eval.*success|evaluation.*success|Success' "$run/runtime/driver.log" 2>/dev/null | tail -n8 || true
  if [[ -s "$run/runtime/resource.csv" ]]; then
    printf 'resource_last='; tail -n1 "$run/runtime/resource.csv"
  fi
done

printf '\n=== RAY ===\n'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_brief_refresh", logging_level="ERROR")
rows=ray.util.list_named_actors(all_namespaces=True)
for ns in ("RLinf","RLinf_1"):
    print(f"{ns}_named_actors={sum(x.get('namespace') == ns for x in rows)}")
ray.shutdown()
PY

printf '\n=== GPU ===\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
for dev in 0 1 2 3 4 5 6 7; do
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    user=$(ps -o user= -p "$pid" | xargs)
    used=$(nvidia-smi -i "$dev" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v p="$pid" '$1+0==p {gsub(/ /,"",$2); print $2; exit}')
    job=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    cmd=$(ps -o args= -p "$pid" | cut -c1-120)
    printf 'gpu=%s pid=%s user=%s used_mib=%s job=%s cmd=%s\n' "$dev" "$pid" "$user" "${used:-?}" "${job:-none}" "$cmd"
  done < <(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
done

printf '\n=== HOST ===\n'
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
df -hT / /home /data | awk 'NR==1 || !seen[$7]++'
printf 'failed_units='; systemctl --failed --no-legend 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l
printf 'mihomo='; systemctl is-active mihomo 2>/dev/null || true
printf 'proxy_7890_listeners='; ss -lnt 2>/dev/null | awk '$4 ~ /127\.0\.0\.1:7890$/ {n++} END {print n+0}'
printf 'github_proxy='
curl -x http://127.0.0.1:7890 -I -L --max-time 6 -o /dev/null -sS -w 'http=%{http_code} time=%{time_total}s\n' https://github.com/ || echo failed
printf 'hf_proxy='
curl -x http://127.0.0.1:7890 -I -L --max-time 6 -o /dev/null -sS -w 'http=%{http_code} time=%{time_total}s\n' https://huggingface.co/ || echo failed

printf '\n=== USER_SUMMARY ===\n'
ps -eo user=,rss=,pcpu= --no-headers | awk '{rss[$1]+=$2; cpu[$1]+=$3; n[$1]++} END {for (u in n) if (u!="root" || rss[u]>1048576) printf "%s processes=%d rss_gib=%.2f cpu_pct=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort -k3,3nr
