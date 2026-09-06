#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
stat -c 'driver_log size=%s mtime=%y' "$RUN/runtime/driver.log"
echo 'last_progress:'
grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint|save_checkpoint' "$RUN/runtime/driver.log" | tail -n 12 || true
echo 'checkpoint_dirs:'
find "$RUN" -type d -name 'global_step_10' -print -exec du -sh {} \; 2>/dev/null || true
echo 'checkpoint_recent_files:'
find "$RUN" -path '*/global_step_10/*' -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 20 || true
echo 'checkpoint_workers:'
for pid in $(pgrep -u chenyiteng -f 'ray::EmbodiedFS' || true); do
  args=$(ps -o args= -p "$pid")
  [[ "$args" == *save_checkpoint* ]] || continue
  printf 'pid=%s etimes=%s cpu=%s state=%s wchan=%s\n' "$pid" \
    "$(ps -o etimes= -p "$pid" | xargs)" "$(ps -o %cpu= -p "$pid" | xargs)" \
    "$(ps -o stat= -p "$pid" | xargs)" "$(ps -o wchan= -p "$pid" | xargs)"
  awk '/read_bytes:|write_bytes:|cancelled_write_bytes:/{print "  "$0}' "/proc/$pid/io" 2>/dev/null || true
done
