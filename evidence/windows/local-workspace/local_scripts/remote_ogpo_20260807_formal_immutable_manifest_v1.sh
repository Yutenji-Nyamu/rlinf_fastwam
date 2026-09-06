#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
files=(
  source_config.yaml
  resolved.yaml
  exact_command.txt
  run_provenance.tsv
  stop_conditions.txt
  resources_before.txt
  launched_at.txt
  started_at.txt
  driver_pid.txt
  monitor_pid.txt
)

printf 'MANIFEST_AT\t%s\n' "$(date --iso-8601=seconds)"
for name in "${files[@]}"; do
  path="$runtime_root/$name"
  test -f "$path"
  printf '%s\t%s\t%s\n' "$(sha256sum "$path" | awk '{print $1}')" \
    "$(stat --printf='%s' "$path")" "$name"
done
