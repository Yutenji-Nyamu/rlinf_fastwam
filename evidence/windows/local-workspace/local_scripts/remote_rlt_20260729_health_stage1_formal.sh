#!/usr/bin/env bash
set -u
export LC_ALL=C

RUNTIME=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/runtime
LOG=${RUNTIME}/driver.log
CSV=${RUNTIME}/resources.csv
driver_pid=$(cat "$RUNTIME/driver_pid.txt" 2>/dev/null || printf '%s' -1)

printf 'OBSERVED_AT\t%s\n' "$(date --iso-8601=seconds)"
if kill -0 "$driver_pid" 2>/dev/null; then
  printf 'STATE\tRUNNING\n'
else
  printf 'STATE\tFINISHED\n'
fi
if test -f "$RUNTIME/exit_code.txt"; then
  printf 'EXIT_CODE\t%s\n' "$(cat "$RUNTIME/exit_code.txt")"
fi
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
printf 'WORKER_COUNTS\n'
printf 'fsdp_init\t%s\n' "$(pgrep -fc 'FSDPVlaSftWorker.init_worker' || true)"
printf 'actor_group\t%s\n' "$(pgrep -fc 'ray::ActorGroup' || true)"
printf 'ERROR_COUNT\t'
grep -Eic 'out of memory|CUDA error|Traceback|NCCL.*error|ChildFailed|killed' \
  "$LOG" 2>/dev/null || printf '0\n'
printf 'METRICS\n'
grep -a 'Global Step:' "$LOG" 2>/dev/null | tr '\r' '\n' | \
  grep 'train/loss=' | tail -n 12 | cut -c1-1200 || true
printf 'HEALTH_SIGNALS\n'
grep -aE 'local_batch_size: 16|Total trainable params|rlt_module|Disabled gradient checkpointing|Starting training|Loaded norm stats' \
  "$LOG" 2>/dev/null | tail -n 20 | cut -c1-800 || true
printf 'RESOURCE_SUMMARY\n'
if test -f "$CSV"; then
  awk -F, '
    NR == 2 || $6 > max_gpu0 {max_gpu0=$6}
    NR == 2 || $8 > max_gpu1 {max_gpu1=$8}
    NR == 2 || $10 > max_rss {max_rss=$10}
    END {
      printf "rows=%d max_gpu0_mib=%s max_gpu1_mib=%s max_matched_rss_kib=%s\n",
             NR, max_gpu0+0, max_gpu1+0, max_rss+0
    }
  ' "$CSV"
  tail -n 2 "$CSV"
fi
