#!/usr/bin/env bash
set -u

chain=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2
stage2=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3

date --iso-8601=seconds
printf '%s\n' '--- chain root files within depth 5 ---'
find "$chain" -maxdepth 5 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 160 || true
printf '%s\n' '--- all candidate logs and events ---'
find "$chain" "$stage2" -type f \
  \( -name '*.log' -o -name '*.csv' -o -name 'events.out.tfevents*' \) \
  -printf '%s %p\n' 2>/dev/null | sort -n || true

printf '%s\n' '--- stage2 metrics head/tail ---'
head -n 4 "$stage2/metrics.log" 2>/dev/null || true
tail -n 40 "$stage2/metrics.log" 2>/dev/null || true

printf '%s\n' '--- stage1 likely metrics head/tail ---'
while IFS= read -r file; do
  printf '### %s\n' "$file"
  head -n 3 "$file" 2>/dev/null || true
  tail -n 12 "$file" 2>/dev/null || true
done < <(find "$chain" -type f -name 'metrics.log' 2>/dev/null | sort)

