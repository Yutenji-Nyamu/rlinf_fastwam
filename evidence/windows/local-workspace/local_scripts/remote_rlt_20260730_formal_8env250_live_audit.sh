#!/usr/bin/env bash
set -u

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1
driver="$(cat "$runtime/driver_pid.txt")"
monitor="$(cat "$runtime/monitor_pid.txt")"

printf 'time\t%s\n' "$(date --iso-8601=seconds)"
printf 'driver\t%s\t%s\n' "$driver" "$(kill -0 "$driver" 2>/dev/null && echo alive || echo exited)"
printf 'monitor\t%s\t%s\n' "$monitor" "$(kill -0 "$monitor" 2>/dev/null && echo alive || echo exited)"
printf 'exit_code\t%s\n' "$(cat "$runtime/exit_code.txt" 2>/dev/null || echo pending)"
printf 'started_at\t%s\n' "$(cat "$runtime/started_at.txt" 2>/dev/null || echo pending)"
printf 'driver_bytes\t%s\n' "$(stat -c %s "$runtime/driver.log" 2>/dev/null || echo 0)"
printf 'resource_rows\t%s\n' "$(wc -l <"$runtime/resources.csv" 2>/dev/null || echo 0)"
printf 'gpu_begin\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'gpu_end\n'
printf 'cgroup_current\t%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'cgroup_anon\t%s\n' "$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat)"
printf 'cgroup_file\t%s\n' "$(awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat)"
printf 'memory_events\t%s\n' "$(tr '\n' ' ' </sys/fs/cgroup/memory.events)"
printf 'memory_psi\t%s\n' "$(tr '\n' ' ' </proc/pressure/memory)"
printf 'resource_tail_begin\n'
tail -n 3 "$runtime/resources.csv" 2>/dev/null || true
printf 'resource_tail_end\n'
printf 'progress_begin\n'
grep -E \
  'episode_len=|num_trajectories=|cache_size=|actor_switch_rate|global_min_replay|global_total_transitions|update_step=|updates_to_run=|Saving checkpoint|eval/|Error|Traceback|OOM|NaN|NCCL' \
  "$runtime/driver.log" 2>/dev/null | tail -n 100 || true
printf 'progress_end\n'
printf 'run_files\t%s\n' "$(
  find "$run_root" -type f 2>/dev/null | wc -l
)"
