#!/usr/bin/env bash
set -u
export LC_ALL=C

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1/runtime
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1
experiment=robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1
driver="$(cat "$runtime/driver_pid.txt" 2>/dev/null || true)"
monitor="$(cat "$runtime/monitor_pid.txt" 2>/dev/null || true)"

printf 'NOW\t%s\n' "$(date --iso-8601=seconds)"
printf 'DRIVER\t%s\t%s\n' "$driver" "$(
  test -n "$driver" && kill -0 "$driver" 2>/dev/null && echo alive || echo exited
)"
printf 'MONITOR\t%s\t%s\n' "$monitor" "$(
  test -n "$monitor" && kill -0 "$monitor" 2>/dev/null && echo alive || echo exited
)"
printf 'EXIT_CODE\t%s\n' "$(cat "$runtime/exit_code.txt" 2>/dev/null || echo pending)"
printf 'STARTED_AT\t%s\n' "$(cat "$runtime/started_at.txt" 2>/dev/null || echo pending)"
printf 'DRIVER_BYTES\t%s\n' "$(stat -c %s "$runtime/driver.log" 2>/dev/null || echo 0)"
printf 'RESOURCE_ROWS\t%s\n' "$(wc -l <"$runtime/resources.csv" 2>/dev/null || echo 0)"
printf 'PROCESS_BEGIN\n'
if test -n "$driver"; then
  ps -o pid=,ppid=,etime=,stat=,rss=,cmd= -p "$driver" || true
fi
pgrep -ax raylet || true
pgrep -ax gcs_server || true
printf 'PROCESS_END\n'
printf 'GPU_BEGIN\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'GPU_END\n'
printf 'CGROUP_CURRENT\t%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'CGROUP_ANON\t%s\n' "$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat)"
printf 'CGROUP_FILE\t%s\n' "$(awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat)"
printf 'MEMORY_EVENTS\t%s\n' "$(tr '\n' ' ' </sys/fs/cgroup/memory.events)"
printf 'MEMORY_PSI\t%s\n' "$(tr '\n' ' ' </proc/pressure/memory)"
printf 'TMP_AVAIL_BYTES\t%s\n' "$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)"
printf 'RESOURCE_TAIL_BEGIN\n'
tail -n 3 "$runtime/resources.csv" 2>/dev/null || true
printf 'RESOURCE_TAIL_END\n'
printf 'PROGRESS_BEGIN\n'
grep -E \
  'Resuming training|Loading|Restor|Global Step|episode_len=|num_trajectories=|cache_size=|actor_switch_rate|global_min_replay|global_total_transitions|update_step=|updates_to_run=|rlt/ready|rlt/ramp|Saving checkpoint|eval/|Error|Traceback|OOM|NaN|NCCL|RuntimeError|ValueError' \
  "$runtime/driver.log" 2>/dev/null | tail -n 180 || true
printf 'PROGRESS_END\n'
printf 'LOG_TAIL_BEGIN\n'
tail -n 80 "$runtime/driver.log" 2>/dev/null || true
printf 'LOG_TAIL_END\n'
printf 'RUN_FILES\t%s\n' "$(
  find "$run_root" -type f 2>/dev/null | wc -l
)"
printf 'CHECKPOINTS\t%s\n' "$(
  find "${run_root}/${experiment}/checkpoints" \
    -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
    | sort | tr '\n' ' '
)"
