#!/usr/bin/env bash
set -euo pipefail

uid=$(id -u)
printf 'TIME=%s\n' "$(date --iso-8601=seconds)"
for pattern in '^ray::EmbodiedFSDPActor' '^ray::MultiStepRolloutWorker' '^ray::EnvWorker'; do
  count=$(pgrep -u "$uid" -f "$pattern" | wc -l)
  printf 'PATTERN=%s COUNT=%s\n' "$pattern" "$count"
  pgrep -a -u "$uid" -f "$pattern" || true
done
printf 'GCS_COUNT=%s\n' "$(pgrep -u "$uid" -x gcs_server | wc -l)"
printf 'RAYLET_COUNT=%s\n' "$(pgrep -u "$uid" -x raylet | wc -l)"
pgrep -a -u "$uid" -x gcs_server || true
pgrep -a -u "$uid" -x raylet || true
