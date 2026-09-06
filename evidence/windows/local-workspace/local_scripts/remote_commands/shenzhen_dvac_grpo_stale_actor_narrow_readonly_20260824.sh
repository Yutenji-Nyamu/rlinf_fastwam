#!/usr/bin/env bash
set -u

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

echo '=== time ==='
date '+%F %T %Z'

declare -A rss_kib=()
declare -A pss_kib=()
declare -A gpu_mib=()
declare -A count=()

mapfile -t rows < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
for row in "${rows[@]}"; do
  pid=${row%%,*}; pid=${pid//[[:space:]]/}
  mem=${row##*,}; mem=${mem//[[:space:]]/}
  [[ -r "/proc/$pid/environ" ]] || continue
  env_dump=$(tr '\0' '\n' < "/proc/$pid/environ")
  job=$(sed -n 's/^RAY_JOB_ID=//p' <<<"$env_dump" | head -n1)
  cuda=$(sed -n 's/^CUDA_VISIBLE_DEVICES=//p' <<<"$env_dump" | head -n1)
  [[ -n "$job" ]] || job=nonray
  rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
  pss=$(awk '/^Pss:/{print $2}' "/proc/$pid/smaps_rollup" 2>/dev/null || echo 0)
  cmd=$(ps -o comm= -p "$pid" 2>/dev/null | xargs)
  printf 'pid=%s job=%s cuda=%s gpu_mib=%s rss_kib=%s pss_kib=%s cmd=%s\n' "$pid" "$job" "$cuda" "$mem" "$rss" "$pss" "$cmd"
  rss_kib[$job]=$(( ${rss_kib[$job]:-0} + rss ))
  pss_kib[$job]=$(( ${pss_kib[$job]:-0} + pss ))
  gpu_mib[$job]=$(( ${gpu_mib[$job]:-0} + mem ))
  count[$job]=$(( ${count[$job]:-0} + 1 ))
done

echo '=== totals by Ray job among GPU processes ==='
for job in "${!count[@]}"; do
  awk -v job="$job" -v n="${count[$job]}" -v gpu="${gpu_mib[$job]}" -v rss="${rss_kib[$job]}" -v pss="${pss_kib[$job]}" \
    'BEGIN {printf "job=%s processes=%d gpu_mib=%d rss_gib=%.3f pss_gib=%.3f\n", job,n,gpu,rss/1048576,pss/1048576}'
done | sort

echo '=== old job actor records ==='
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" list actors --filter 'job_id=1b000000' --detail 2>&1 | sed -n '1,500p' || true

echo '=== candidate new job actor records ==='
for job in 1c000000 1d000000 1e000000 1f000000 20000000; do
  out=$(RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" list actors --filter "job_id=$job" --detail 2>&1 || true)
  if grep -qE '^-' <<<"$out"; then
    echo "job=$job"
    sed -n '1,500p' <<<"$out"
  fi
done

echo '=== current wrapper and progress ==='
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3
pid=$(<"$RUN/runtime/wrapper.pid")
if kill -0 "$pid" 2>/dev/null; then echo "current_wrapper_alive=yes pid=$pid"; else echo "current_wrapper_alive=no pid=$pid"; fi
grep -E 'Global Step:|Generating Rollout Epochs:' "$RUN/runtime/driver.log" | tail -n 8 || true
printf 'fatal_matches='; grep -Eci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$RUN/runtime/driver.log" || true

