#!/usr/bin/env bash
set -euo pipefail
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
for f in run_foreground.sh resource_monitor.sh prepare_formal.sh launch_detached.sh; do
  p="$runtime/$f"
  if [ -f "$p" ]; then
    echo "===== $f ====="
    sed -n '1,260p' "$p"
  fi
done
