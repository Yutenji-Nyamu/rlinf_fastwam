#!/usr/bin/env bash
set -u

run_root=/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823
phase=${1:-fresh}
log_dir=$run_root/$phase
wrapper_pid=$(cat "$log_dir/wrapper.pid")

while kill -0 "$wrapper_pid" 2>/dev/null; do
  printf '\nWATCH %s phase=%s wrapper=%s\n' "$(date --iso-8601=seconds)" "$phase" "$wrapper_pid"
  free -h | awk 'NR == 1 || NR == 2 {print}'
  nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  tail -n 12 "$log_dir/driver.log" 2>/dev/null || true
  sleep 30
done

printf '\nWATCH_END %s phase=%s wrapper=%s\n' "$(date --iso-8601=seconds)" "$phase" "$wrapper_pid"
if test -f "$log_dir/exit_code.txt"; then
  printf 'exit_code=%s\n' "$(cat "$log_dir/exit_code.txt")"
else
  printf '%s\n' 'exit_code=missing'
fi
free -h | awk 'NR == 1 || NR == 2 {print}'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 80 "$log_dir/driver.log" 2>/dev/null || true
