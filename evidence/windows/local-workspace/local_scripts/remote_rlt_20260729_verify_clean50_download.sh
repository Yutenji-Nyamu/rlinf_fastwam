#!/usr/bin/env bash
set -euo pipefail

revision=9dc9299c163db059931898a9f0852098a61155a1
root=/root/autodl-tmp/datasets/robotwin2/source/${revision}
target=${root}/dataset/adjust_bottle/aloha-agilex_clean_50.zip
log=/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.log

echo "VERIFY_TIME $(date -Is)"
stat --printf='path=%n\nbytes=%s\nmtime=%y\n' "$target"
sha256sum "$target"
unzip -tqq "$target"
echo "zip_test=PASS"

entries=$(unzip -Z1 "$target" | wc -l)
traj_count=$(unzip -Z1 "$target" \
  | grep -Ec '^aloha-agilex_clean_50/_traj_data/episode[0-9]+\.pkl$')
camera_count=$(unzip -Z1 "$target" \
  | grep -Ec '^aloha-agilex_clean_50/.*/episode[0-9]+\.(mp4|avi)$' || true)
instruction_count=$(unzip -Z1 "$target" \
  | grep -Ec '^aloha-agilex_clean_50/instructions/episode[0-9]+\.json$')
echo "archive_entries=$entries"
echo "trajectory_episode_pkls=$traj_count"
echo "camera_episode_videos=$camera_count"
echo "instruction_episode_json=$instruction_count"
unzip -l "$target" | tail -1

echo "=== producer_log_summary ==="
grep -E \
  '^(START|network_path|hf_home|actual_size|actual_sha256|archive_entries|archive_uncompressed_bytes|SUCCESS)' \
  "$log"

echo "=== source_root_files ==="
find "$root" \
  -maxdepth 6 \
  -type f \
  -printf '%p|%s|%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' \
  | sort
du -x -B1 --max-depth=5 "$root" | sort -nr | head -20
df -h /root/autodl-tmp

echo "=== relevant_processes ==="
pgrep -af \
  'rlt_clean50_download|hf_hub_download|aloha-agilex_clean_50' \
  || true
