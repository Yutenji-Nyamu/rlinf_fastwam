#!/usr/bin/env bash
set -u

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
ACTION="$ROOT/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"
ST="$ROOT/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1"
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

printf 'NOW=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
for item in "ACTION_MID:$ACTION" "ST_NARROW:$ST"; do
  label=${item%%:*}; run=${item#*:}
  printf '\n=== %s ===\n' "$label"
  pid=$(cat "$run/runtime/wrapper.pid" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then alive=1; else alive=0; fi
  printf 'pid=%s alive=%s exit=' "$pid" "$alive"
  cat "$run/runtime/exit_code.txt" 2>/dev/null || echo pending
  printf 'fatal_count='; grep -aEic 'Traceback|CUDA out of memory|Error executing job|RayActorError|WorkerCrashedError|NCCL.*(error|failed)|non.?finite' "$run/runtime/driver.log" 2>/dev/null || true
  printf 'latest_complete='; grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$run/runtime/driver.log" 2>/dev/null | tail -n1 || echo none
  grep -aE 'Creating (rollout|env|actor) worker|Generating Rollout Epochs:|Global Step:|Saving checkpoint' "$run/runtime/driver.log" 2>/dev/null | tail -n4 || true
  printf 'driver_bytes='; stat -c %s "$run/runtime/driver.log" 2>/dev/null || echo 0
done

printf '\n=== NAMESPACES ===\n'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_new_pair_startup", logging_level="ERROR")
rows=ray.util.list_named_actors(all_namespaces=True)
for ns in ("RLinf", "RLinf_1"):
    print(f"{ns}={sum(x.get('namespace') == ns for x in rows)}")
ray.shutdown()
PY

printf '\n=== GPU4-7 ===\n'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
for dev in 4 5 6 7; do
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    printf 'gpu=%s pid=%s user=%s cmd=%s\n' "$dev" "$pid" "$(ps -o user= -p "$pid" | xargs)" "$(ps -o args= -p "$pid" | cut -c1-100)"
  done < <(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
done

printf '\n=== HOST ===\n'
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
df -hT / /home /data | awk 'NR==1 || !seen[$7]++'
printf 'failed_units='; systemctl --failed --no-legend 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l
printf 'proxy='; curl -x http://127.0.0.1:7890 -I -L --max-time 6 -o /dev/null -sS -w 'github=%{http_code} %{time_total}s\n' https://github.com/ || echo failed

printf '\n=== USERS ===\n'
ps -eo user=,rss=,pcpu= --no-headers | awk '{rss[$1]+=$2; cpu[$1]+=$3; n[$1]++} END {for (u in n) if (rss[u]>1048576 || cpu[u]>5) printf "%s processes=%d rss_gib=%.2f cpu_pct=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort
