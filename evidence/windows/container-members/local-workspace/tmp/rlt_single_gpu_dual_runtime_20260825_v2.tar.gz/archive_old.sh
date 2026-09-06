#!/usr/bin/env bash
set -euo pipefail

old_run=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
old_runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
export_root=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/high_info_archive_20260825
stage="$export_root/staging"
archive="$export_root/rlt_teacher_dvac_w0to2_formal_fresh480_high_info_raw_20260825.tar.gz"

if [ -e "$stage" ] || [ -e "$archive" ]; then
  echo "archive target already exists: $export_root" >&2
  exit 2
fi
mkdir -p "$stage"

# Runtime evidence is small and complete: command, resolved config, logs, resources,
# lifecycle markers and hashes.
find "$old_runtime" -type f -size -32M -print0 | xargs -0 -r cp --parents -t "$stage"

# Main run evidence excluding large checkpoint payloads and video.
find "$old_run" -type f \
  ! -path '*/checkpoints/*' \
  ! -name '*.mp4' ! -name '*.avi' \
  -size -32M -print0 | xargs -0 -r cp --parents -t "$stage"

# Preserve checkpoint structure/commit evidence without model, optimizer or replay bodies.
find "$old_run/checkpoints" -type f \
  \( -name 'checkpoint_complete.json' -o -name 'rlt_trainer_state_complete.json' -o -name '*manifest*.json' -o -name '*inventory*.txt' \) \
  -size -2M -print0 | xargs -0 -r cp --parents -t "$stage"

{
  echo "source_head=$(git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac rev-parse HEAD)"
  echo "archive_created=$(date -Is)"
  echo "old_exit_code=$(cat "$old_runtime/exit_code.txt")"
  echo "old_finished_at=$(cat "$old_runtime/finished_at.txt")"
  echo 'Large checkpoint tensors, optimizer/replay bodies, videos and Ray session logs are intentionally excluded.'
} >"$stage/ARCHIVE_README.txt"

find "$old_run" -type f -printf '%s\t%p\n' | sort -nr >"$stage/OLD_RUN_FULL_FILE_INVENTORY.txt"
du -ah "$old_run/checkpoints" 2>/dev/null | sort -h >"$stage/CHECKPOINT_SIZE_INVENTORY.txt"

mkdir -p "$export_root"
tar -C "$stage" -czf "$archive" .
sha256sum "$archive" >"${archive}.sha256"
du -h "$archive" | tee "$export_root/archive_size.txt"
printf '%s\n' "$archive" >"$export_root/archive_path.txt"
