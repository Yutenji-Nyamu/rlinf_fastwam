#!/usr/bin/env bash
set -euo pipefail
runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v1/runtime
launcher=$(cat /tmp/rlt_dvac_pure_smoke_launcher_20260827_v1.pid)
ray_head=$(cat "$runtime/ray_head.pid")
driver=$(cat "$runtime/driver.pid")
for label_pid in launcher:$launcher ray_head:$ray_head driver:$driver; do
  label=${label_pid%%:*}; pid=${label_pid#*:}
  kill -0 "$pid" 2>/dev/null && echo "$label=$pid alive" || echo "$label=$pid stopped"
done
test -f "$runtime/exit_code.txt" && { printf 'exit='; cat "$runtime/exit_code.txt"; }
pgrep -af 'rlt_dvac_pure_smoke_v1|ray_rlt_dvac_pure_smoke_v1' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
