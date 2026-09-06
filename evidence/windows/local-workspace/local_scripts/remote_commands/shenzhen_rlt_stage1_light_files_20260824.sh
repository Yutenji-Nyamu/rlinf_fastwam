#!/usr/bin/env bash
set -u

root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2

date --iso-8601=seconds
find "$root" -maxdepth 4 -type f \
  \( -name '*.log' -o -name '*.csv' -o -name '*.yaml' -o -name '*.txt' -o -name 'events.out.tfevents.*' \) \
  -printf '%s|%TY-%Tm-%TdT%TH:%TM:%TS|%p\n' 2>/dev/null | sort
printf '%s\n' '--- top tree ---'
find "$root" -maxdepth 3 -mindepth 1 -printf '%y|%s|%p\n' 2>/dev/null | sort
