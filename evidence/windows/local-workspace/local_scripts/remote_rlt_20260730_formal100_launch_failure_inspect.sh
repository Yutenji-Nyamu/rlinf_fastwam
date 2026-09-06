#!/usr/bin/env bash
set -u
export LC_ALL=C

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime
run=/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1

printf 'inspect_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'run_exists\t%s\n' "$(test -e "${run}" && printf yes || printf no)"
for name in driver_pid.txt monitor_pid.txt started_at.txt exit_code.txt finished_at.txt driver.log resources.csv; do
  printf '%s\t%s\n' "${name}" "$(
    test -e "${runtime}/${name}" \
      && stat -c 'EXISTS size=%s' "${runtime}/${name}" \
      || printf MISSING
  )"
done
printf 'processes_begin\n'
pgrep -af \
  'train_embodied_agent.py|rlt_stage2_formal_100c|raylet|gcs_server' \
  || true
printf 'processes_end\n'
printf 'gpu_begin\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
printf 'gpu_end\n'
printf 'launch_script_begin\n'
sed -n '1,220p' "${runtime}/launch_background.sh"
printf 'launch_script_end\n'
