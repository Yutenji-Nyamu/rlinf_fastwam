#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
ACTION="$ROOT/dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"
ST="$ROOT/dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1"
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
printf 'NOW=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
for item in "action:$ACTION" "st:$ST"; do
  label=${item%%:*}; run=${item#*:}
  printf '\n=== %s ===\n' "$label"
  pid=$(cat "$run/runtime/wrapper.pid" 2>/dev/null || true)
  printf 'pid=%s alive=' "$pid"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo 1; else echo 0; fi
  printf 'exit='; cat "$run/runtime/exit_code.txt" 2>/dev/null || echo pending
  printf 'fatal_count='
  grep -aEic 'Traceback|CUDA out of memory|Error executing job|RayActorError|WorkerCrashedError|NCCL.*(error|failed)|non.?finite' "$run/runtime/driver.log" 2>/dev/null || true
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100|Rollout Epoch:[[:space:]]+[0-9]+/4|Initializing [A-Za-z]+Manager|RAY_NAMESPACE[^ ]*|ERROR[^[:cntrl:]]*' "$run/runtime/driver.log" 2>/dev/null | tail -n 12 || true
  tail -n 8 "$run/runtime/wrapper.log" 2>/dev/null || true
done

printf '\n=== namespaces ===\n'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_startup_snapshot", logging_level="ERROR")
rows=ray.util.list_named_actors(all_namespaces=True)
for ns in ("RLinf","RLinf_1"):
    names=sorted(x["name"] for x in rows if x.get("namespace")==ns)
    print(ns, len(names), ",".join(names))
ray.shutdown()
PY

printf '\n=== gpu ===\n'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
for dev in 4 5 6 7; do
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    user=$(ps -o user= -p "$pid" | xargs)
    job=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    printf 'gpu=%s pid=%s user=%s job=%s cmd=' "$dev" "$pid" "$user" "$job"
    ps -o args= -p "$pid" | cut -c1-160
  done < <(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
done
printf '\nMemAvailable='; awk '/^MemAvailable:/ {print $2" "$3}' /proc/meminfo
