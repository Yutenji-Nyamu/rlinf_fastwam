set -u
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'DVAC_REFERENCES\n'
grep -RInE "rlt_dvac|teacher_dvac|dvac_telemetry|return_dvac" "$repo/rlinf" "$repo/examples/embodiment" | head -240
printf 'EVAL_CALLS\n'
grep -RInE "evaluate|validation|val_check|eval_only|generate_rollout" "$repo/rlinf/runners" "$repo/rlinf/algorithms" | head -240
printf 'RUNNER_EVAL_SNIPPET\n'
sed -n '185,215p' "$repo/rlinf/runners/embodied_runner.py"
printf 'MODEL_DVAC_SNIPPET\n'
sed -n '660,755p' "$repo/rlinf/models/embodiment/openpi/openpi_action_model.py"
printf 'ROLLOUT_EVAL_SNIPPET\n'
grep -RIn "def evaluate" "$repo/rlinf/workers/rollout" | head -20
