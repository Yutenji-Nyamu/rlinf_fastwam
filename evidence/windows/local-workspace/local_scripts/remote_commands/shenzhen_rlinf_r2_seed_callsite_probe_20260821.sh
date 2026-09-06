#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
grep -RIn --include='*.py' 'partition_success_seeds' "$ROOT/rlinf" | while IFS=: read -r file line rest; do
  printf '%s:%s:%s\n' "$file" "$line" "$rest"
  start=$((line > 50 ? line - 50 : 1))
  end=$((line + 80))
  nl -ba "$file" | sed -n "${start},${end}p"
done
printf '%s\n' '=== OFFICIAL EVAL YAML ==='
nl -ba "$ROOT/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml" | sed -n '1,260p'
