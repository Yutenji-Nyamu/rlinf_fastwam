#!/usr/bin/env bash

base=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2/robotwin_adjust_bottle_ogpo_ca_formal_90k_utd005_v2/checkpoints
for step in 22 43; do
  manifest="$base/global_step_$step/actor/ogpo_components/complete.json"
  printf '\n## global_step_%s\n' "$step"
  stat -c '%n | %s bytes | %y' "$manifest" 2>&1
  sed -n '1,200p' "$manifest" 2>&1
done
