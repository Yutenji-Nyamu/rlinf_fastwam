#!/usr/bin/env bash
set -euo pipefail

NEW_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2
ACTION_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

for label in action st; do
  if [[ "$label" = action ]]; then run=$ACTION_RUN; else run=$NEW_RUN; fi
  pid=$(cat "$run/runtime/wrapper.pid")
  printf '%s_wrapper_pid=%s alive=' "$label" "$pid"
  if kill -0 "$pid" 2>/dev/null; then echo yes; else echo no; fi
  printf '%s_log_size=%s mtime=%s\n' "$label" "$(stat -c %s "$run/runtime/driver.log")" "$(stat -c %y "$run/runtime/driver.log")"
  printf '%s_fatal=%s\n' "$label" "$(grep -aEic 'Traceback|OutOfMemory|CUDA out of memory|nonfinite|nan loss|ErrorInitializationFailed' "$run/runtime/driver.log" || true)"
  printf '%s_progress:\n' "$label"
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100|Rollout Epoch:[[:space:]]+[0-9]+/[0-9]+' "$run/runtime/driver.log" | tail -n8 || true
  if [[ "$label" = st ]]; then
    printf 'st_tail:\n'
    tail -n20 "$run/runtime/driver.log"
  fi
done

RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ['RAY_ADDRESS'], namespace='codex_st_v2_startup_readonly', logging_level='ERROR')
for ns in ('RLinf','RLinf_1'):
    rows=[r for r in ray.util.list_named_actors(all_namespaces=True) if r.get('namespace')==ns]
    print(f'{ns}_actors={len(rows)} states=' + ','.join(sorted({str(r.get("state")) for r in rows})))
ray.shutdown()
PY

nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
for devices in 4,5 6,7; do
  printf 'gpu%s_jobs=' "$devices"
  jobs=()
  while read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/environ" ]] || continue
    job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ -n "$job" ]] && jobs+=("$job")
  done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  if ((${#jobs[@]})); then printf '%s\n' "${jobs[@]}" | sort -u | paste -sd, -; else echo none; fi
done
printf 'mem_available_kib=%s\n' "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
