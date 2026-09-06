#!/usr/bin/env bash
set -u
stage=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_stopped_high_info_20260830_v1
archive=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_stopped_high_info_20260830_v1.tar.gz
for path in "$stage" "$archive"; do
  if test -e "$path"; then echo "EXISTS $path"; find "$path" -maxdepth 2 -type f -printf '%s %p\n' | head -30; else echo "MISSING $path"; fi
done
for rt in \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1/runtime \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1/runtime; do
  pid=$(cat "$rt/wrapper.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "WRAPPER $pid alive"; ps -o pid=,pgid=,stat=,args= -p "$pid"; else echo "WRAPPER $pid dead"; fi
done
for run in \
  /root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1 \
  /root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1; do
  echo "RUN $run"
  ls -l "$run/metrics.log" "$run/tensorboard/config.yaml"
  grep -oE 'Global Step: +[0-9]+/480' "$run/metrics.log" | tail -n 1
  find "$run" -maxdepth 6 -type f -name '*.npz' -path '*rlt_dvac*' | tail -n 3
done
