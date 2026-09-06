#!/usr/bin/env bash
set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
printf '=== PRIMARY_FILES ===\n'
for rel in \
  launch_manifest.txt resolved.yaml resolved.yaml.sha256 \
  driver_wrapper.sh driver.pid driver.exit \
  metrics.log resource.csv resource_observer.sh resource_observer.pid resource_observer.log \
  ray_log_snapshot.txt; do
  test -e "$RUN/$rel" && stat -c '%s %y %n' "$RUN/$rel" || printf 'MISSING %s\n' "$RUN/$rel"
done
printf '=== RAY_FINAL_FILES ===\n'
find "$RUN/ray_logs_final" -maxdepth 1 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -n
printf '=== TENSORBOARD_AND_SMALL_METADATA ===\n'
find "$RUN" -maxdepth 4 -type f -size -5M \
  \( -path '*/tensorboard/*' -o -name 'events.out.tfevents*' -o -name '*.yaml' -o -name '*.json' -o -name '*.txt' -o -name '*.sha256' \) \
  -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -n
printf '=== DIRECTORY_TOTALS ===\n'
du -sh "$RUN" "$RUN/ray_logs_final" 2>/dev/null || true
printf 'SZ_GRPO_LIGHT_BUNDLE_INVENTORY_OK\n'
