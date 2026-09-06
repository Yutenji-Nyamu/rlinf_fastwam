#!/usr/bin/env bash
set -euo pipefail

run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1/runtime
checkpoint="$run_root/robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1/checkpoints/global_step_480"

echo "RUNTIME_BEGIN"
find "$runtime" -maxdepth 1 -type f \
  -printf '%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort
echo "RUNTIME_END"
echo "TB_BEGIN"
find "$run_root/tensorboard" -maxdepth 1 -type f \
  -printf '%p\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort
echo "TB_END"
echo "TRACEBACK_CONTEXT_BEGIN"
grep -n -B 8 -A 18 'Traceback (most recent call last)' "$runtime/driver.log" || true
echo "TRACEBACK_CONTEXT_END"
echo "FINAL_COMPLETION_BEGIN"
cat "$checkpoint/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"
echo
echo "FINAL_COMPLETION_END"
