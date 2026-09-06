#!/usr/bin/env bash
set -euo pipefail
printf 'TIME='; date '+%F %T %Z'

for parent in \
  /data/chenyiteng/results/rlinf-shenzhen/*/runs \
  /data/chenyiteng/results/rlinf-rlt \
  /data/chenyiteng/results/rlinf-current-dsrl; do
  test -d "$parent" || continue
  printf '\nPARENT=%s\n' "$parent"
  for run in "$parent"/*; do
    test -d "$run" || continue
    # Ignore container directories under the two non-runs roots.
    base=$(basename "$run")
    case "$base" in packets|launchers|evidence-bundles|exports|smokes) continue;; esac
    exit_value=NA
    for f in "$run"/runtime/exit_code.txt "$run"/exit_code.txt "$run"/fresh/runtime/exit_code.txt; do
      if test -f "$f"; then exit_value=$(tr -d '\r\n ' < "$f"); break; fi
    done
    source_head=NA
    for f in "$run"/source-head.txt "$run"/runtime/source-head.txt "$run"/launch_manifest.txt "$run"/runtime/launch_manifest.txt; do
      test -f "$f" || continue
      if test "$(basename "$f")" = source-head.txt; then source_head=$(head -n1 "$f"); else source_head=$(grep -Eom1 '[0-9a-f]{40}' "$f" || true); fi
      test -n "$source_head" && break
    done
    latest=$(find "$run" -path '*/checkpoints/*' -prune -o -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n1 || true)
    small_bytes=$(find "$run" -path '*/checkpoints/*' -prune -o -path '*/video/*' -prune -o -path '*/videos/*' -prune -o -path '*/robotwin_data/*' -prune -o -type f \
      \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.txt' -o -name '*.jsonl' -o -name 'events.out.tfevents*' -o -name '*.png' \) -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
    printf 'RUN=%s\tEXIT=%s\tSOURCE=%s\tLATEST=%s\tSMALL_BYTES=%s\n' "$run" "$exit_value" "$source_head" "$latest" "$small_bytes"
  done
done
