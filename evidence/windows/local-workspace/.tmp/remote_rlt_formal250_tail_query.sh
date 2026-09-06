#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime
run=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1

printf 'observed_at\t%s\n' "$(date -Iseconds)"
grep -a 'Global Step:' "${runtime}/driver.log" | tail -n 1 || true
tail -n 90 "${runtime}/driver.log"
printf 'checkpoint200\t'
if test -f "${run}/checkpoints/global_step_200/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"; then
  printf 'complete\n'
else
  printf 'absent\n'
fi
