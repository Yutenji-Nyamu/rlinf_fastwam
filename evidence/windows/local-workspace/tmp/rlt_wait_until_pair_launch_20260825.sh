#!/usr/bin/env bash
set -u
old_run=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
queue=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue
last=''
while [ ! -f "$queue/pair_launched_at.txt" ] && [ ! -f "$queue/blocked.txt" ]; do
  step=$(grep -oE 'Global Step:[[:space:]]+[0-9]+/480' "$old_run/metrics.log" 2>/dev/null | tail -n 1 || true)
  if [ "$step" != "$last" ]; then
    printf '%s %s\n' "$(date -Is)" "${step:-old-run-finishing}"
    last="$step"
  fi
  sleep 20
done
echo '[QUEUE_TERMINAL]'
date -Is
for f in blocked.txt old_completed_at.txt pair_launched_at.txt pair_launch_summary.txt archive_old.log; do
  if [ -f "$queue/$f" ]; then
    echo "--$f"
    if [ "$f" = archive_old.log ]; then tail -n 20 "$queue/$f"; else cat "$queue/$f"; fi
  fi
done
