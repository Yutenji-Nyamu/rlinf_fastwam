#!/usr/bin/env bash
set -u

echo '[top_level_bytes]'
timeout 55s du -x -B1 --max-depth=1 /root/autodl-tmp 2>/dev/null | sort -n
echo "top_level_du_exit=${PIPESTATUS[0]}"

echo '[dsrl_worktree_depth2_bytes]'
timeout 40s du -x -B1 --max-depth=2 \
  /root/autodl-tmp/RLinf_fastwam_rlinf 2>/dev/null | sort -n | tail -n 40
echo "dsrl_du_exit=${PIPESTATUS[0]}"

echo '[models_depth2_bytes]'
timeout 40s du -x -B1 --max-depth=2 \
  /root/autodl-tmp/models 2>/dev/null | sort -n | tail -n 40
echo "models_du_exit=${PIPESTATUS[0]}"

echo '[exports_depth2_bytes]'
timeout 30s du -x -B1 --max-depth=2 \
  /root/autodl-tmp/experiment_exports 2>/dev/null | sort -n | tail -n 40
echo "exports_du_exit=${PIPESTATUS[0]}"

echo '[dsrl_large_files]'
find /root/autodl-tmp/RLinf_fastwam_rlinf -xdev -type f -size +1G \
  -printf '%s %p\n' 2>/dev/null | sort -nr | head -n 60

echo '[recent_run_roots]'
find /root/autodl-tmp/RLinf_fastwam_rlinf/logs -mindepth 1 -maxdepth 1 \
  -type d -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -r
