#!/usr/bin/env bash
set -euo pipefail

runtime=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime
run_root=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1
checkpoint="$run_root/robotwin_adjust_bottle_qam_qonly_smoke_20260731_v1/checkpoints/global_step_1"
sidecars="$checkpoint/actor/qam_components"
validator=/root/autodl-tmp/qam_validate_smoke_sidecars.py
python=/root/autodl-tmp/RLinf/.venv/bin/python

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
hostname
pwd
id -u

"$python" "$validator" "$sidecars" >"$runtime/sidecar_validation.json"

{
  echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
  echo "RUN_ROOT=$run_root"
  echo "RUNTIME_ROOT=$runtime"
  echo "RUN_BYTES=$(du -sb "$run_root" | awk '{print $1}')"
  echo "RUNTIME_BYTES=$(du -sb "$runtime" | awk '{print $1}')"
  echo "VIDEO_COUNT=$(find "$run_root" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.webm' \) | wc -l)"
  echo "IMAGE_COUNT=$(find "$run_root" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
  echo "CHECKPOINT_FILES"
  find "$checkpoint" -type f -printf '%s\t%p\n' | sort -nr
  echo "NON_CHECKPOINT_FILES"
  find "$run_root" -path "$checkpoint" -prune -o -type f -printf '%s\t%p\n' | sort -nr
  echo "RUNTIME_FILES"
  find "$runtime" -maxdepth 1 -type f -printf '%s\t%p\n' | sort -nr
} >"$runtime/postrun_inventory.txt"

echo "SIDECAR_CHECKS"
"$python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d["checks"], sort_keys=True))' \
  "$runtime/sidecar_validation.json"
echo "INVENTORY_HEAD"
head -n 16 "$runtime/postrun_inventory.txt"
echo "RESOURCE_ROWS=$(($(wc -l < "$runtime/resources.csv") - 1))"
echo "DRIVER_EXIT=$(cat "$runtime/exit_code.txt")"
echo "ORIGINAL_MONITOR_EXIT=$(cat "$runtime/monitor_exit_code.txt")"
