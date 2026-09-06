#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/fastwam-standalone/fw-sz-400-20260822_042858
VRT=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin
CFG=fw-sz-400-20260822_042858_demo_clean_1ep

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' '=== collect log ==='
sed -n '1,240p' "$RUN/collect_data.log"
printf '%s\n' '=== run and partial outputs ==='
find "$RUN" "$VRT/data/adjust_bottle/$CFG" -maxdepth 4 -printf '%y %s %p\n' 2>/dev/null | sort || true
printf '%s\n' '=== matching processes and gpu ==='
pgrep -af '([c]ollect_data\.py|[t]est_render\.py)' || true
nvidia-smi -i 3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf '%s\n' '=== recent kernel gpu/segfault lines ==='
journalctl -k --since '2026-08-22 12:25:00' --no-pager 2>/dev/null \
  | grep -Ei 'segfault|NVRM|Xid|oom|killed process' | tail -n 80 || true
