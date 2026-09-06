#!/usr/bin/env bash
set -u

printf 'AUDIT_TIME='; date '+%F %T %Z'

printf '\n=== FASTWAM_DIRTY_DETAIL ===\n'
repo=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
git -C "$repo" status --porcelain=v1 --untracked-files=all 2>/dev/null || true
git -C "$repo" diff --stat 2>/dev/null || true

printf '\n=== PI05_CURRENT_SMALL_ARTIFACTS ===\n'
run=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2
if test -d "$run"; then
  printf 'RUN=%s\n' "$run"
  du -sh "$run" 2>/dev/null | sed 's/^/TOTAL=/'
  for sub in runtime tensorboard checkpoints video robotwin_data monitor telemetry figures; do
    test -e "$run/$sub" || continue
    du -sh "$run/$sub" 2>/dev/null | sed "s#^#SUB_${sub}=#"
  done
  find "$run" -type f \
    \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.txt' -o -name '*.jsonl' -o -name 'events.out.tfevents*' -o -name '*.png' \) \
    -printf '%s\t%p\n' 2>/dev/null | sort -nr | sed -n '1,45p'
else
  printf 'MISSING=%s\n' "$run"
fi

printf '\n=== PACKET_SIZES ===\n'
for packet in \
  /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2 \
  /data/chenyiteng/results/rlinf-shenzhen/pi05/packets/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2; do
  test -d "$packet" || continue
  du -sh "$packet"
  find "$packet" -maxdepth 2 -type f -printf '%s\t%p\n' | sort -nr | sed -n '1,20p'
done
