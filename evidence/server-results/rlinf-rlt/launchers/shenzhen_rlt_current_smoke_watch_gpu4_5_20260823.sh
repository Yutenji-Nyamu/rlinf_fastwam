#!/usr/bin/env bash
set -u

phase=${1:-stage1}
case "$phase" in
  stage1)
    log_dir=/data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime
    ;;
  stage2-fresh)
    log_dir=/data/chenyiteng/results/rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh
    ;;
  stage2-resume)
    log_dir=/data/chenyiteng/results/rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume
    ;;
  *)
    printf 'unknown phase: %s\n' "$phase" >&2
    exit 2
    ;;
esac

wrapper_pid=$(cat "$log_dir/wrapper.pid")
while kill -0 "$wrapper_pid" 2>/dev/null; do
  printf '\nWATCH %s phase=%s wrapper=%s\n' "$(date --iso-8601=seconds)" "$phase" "$wrapper_pid"
  free -h | awk 'NR == 1 || NR == 2 {print}'
  nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  tail -n 14 "$log_dir/driver.log" 2>/dev/null || true
  sleep 30
done

printf '\nWATCH_END %s phase=%s wrapper=%s\n' "$(date --iso-8601=seconds)" "$phase" "$wrapper_pid"
if test -f "$log_dir/exit_code.txt"; then
  printf 'exit_code=%s\n' "$(cat "$log_dir/exit_code.txt")"
else
  printf '%s\n' 'exit_code=missing'
fi
free -h | awk 'NR == 1 || NR == 2 {print}'
nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 100 "$log_dir/driver.log" 2>/dev/null || true
