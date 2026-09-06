#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs
CONTROL="$ROOT/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1"
DVAC="$ROOT/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1"
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
RAY_ADDRESS=172.17.0.1:6389

date --iso-8601=seconds
for item in "control:$CONTROL" "dvac:$DVAC"; do
  label=${item%%:*}; run=${item#*:}; pid=$(<"$run/runtime/wrapper.pid")
  echo "=== $label ==="
  echo "pid=$pid alive=$([[ -d /proc/$pid ]] && echo 1 || echo 0) log_bytes=$(stat -c %s "$run/runtime/driver.log")"
  grep -aE 'Using flexible placement|RLinf Ray code sync|Namespace|Generating Rollout Epochs|Global Step:|Traceback|out of memory|CUDA error|ErrorInitializationFailed|worker died|fatal' "$run/runtime/driver.log" | tail -n 16 || true
done

echo '=== namespaces ==='
RAY_ADDRESS="$RAY_ADDRESS" "$PY" - <<'PY'
import collections, ray
ray.init(address="172.17.0.1:6389", namespace="codex_pi05_startup_status", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
print(dict(sorted(collections.Counter(row.get("namespace", "") for row in rows).items())))
ray.shutdown()
PY

echo '=== gpu ==='
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
for gpu in 4 5 6 7; do
  while read -r pid; do
    [[ -n "$pid" && -r /proc/$pid/environ ]] || continue
    job=$(tr '\0' '\n' < /proc/$pid/environ | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    user=$(ps -o user= -p "$pid" | xargs)
    code=$(tr '\0' '\n' < /proc/$pid/environ | sed -n 's/^RLINF_CODE_WORKING_DIR=//p' | head -n1)
    echo "gpu=$gpu pid=$pid user=$user job=$job code=$code"
  done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
done
echo '=== memory ==='
awk '/^MemTotal:|^MemAvailable:|^SwapTotal:|^SwapFree:/' /proc/meminfo
cat /proc/pressure/memory
