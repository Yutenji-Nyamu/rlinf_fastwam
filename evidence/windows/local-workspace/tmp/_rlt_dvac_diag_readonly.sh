set -u
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime
printf 'DRIVER_SUCCESS_REWARD_ROUTE\n'
grep -anEi 'success|reward|actor_switch|rollout|training|eval|validation|run_training|metrics' "$runtime/driver.log" | tail -250
printf 'TRACE_FILES\n'
find "$runtime" -maxdepth 4 -type f -printf '%p %s\n' | sort