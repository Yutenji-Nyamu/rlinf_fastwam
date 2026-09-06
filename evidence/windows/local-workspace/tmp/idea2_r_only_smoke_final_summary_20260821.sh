#!/usr/bin/env bash
set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821

for rank in 00 01; do
  echo "RANK_${rank}_METRICS"
  tail -n 2 "$run/dvac_train/actor_rank${rank}/runner_step_metrics.csv"
done

echo TRAIN_METRICS
grep -aE 'Global Step:|success_once=|actor/approx_kl=|actor/clip_fraction=|actor/dvac_|actor/grad_norm=' "$run/metrics.log" | tail -n 24 || true

echo CHECKPOINTS
find "$run" -type d -name 'global_step_*' -printf '%p\n' 2>/dev/null | sort -V

echo CONTROL_TRACE
find "$run/control_trace" -type f -printf '%p %s bytes\n' 2>/dev/null | sort

echo MEMORY_EVENTS
cat /sys/fs/cgroup/memory.events

echo FINAL_GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

echo RESOURCE_ROWS
wc -l "$runtime/resource_monitor/resources.csv"
