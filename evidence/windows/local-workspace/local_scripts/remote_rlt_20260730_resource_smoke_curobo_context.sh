#!/usr/bin/env bash
set -u

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime
driver="$(cat "$runtime/driver_pid.txt")"
printf 'time\t%s\n' "$(date --iso-8601=seconds)"
printf 'driver\t%s\n' "$(kill -0 "$driver" 2>/dev/null && echo alive || echo exited)"
printf 'exit\t%s\n' "$(cat "$runtime/exit_code.txt" 2>/dev/null || echo pending)"
printf 'tail_begin\n'
tail -n 260 "$runtime/driver.log"
printf 'tail_end\n'
printf 'ray_processes_begin\n'
ps -eo pid=,ppid=,comm=,rss=,args= \
  | grep -E 'ray::|raylet|gcs_server|default_worker.py|train_embodied_agent' \
  | grep -v grep || true
printf 'ray_processes_end\n'
