set -u
runtime=/root/autodl-tmp/experiment_exports/qam_prompt_fixcheck_20260801_v1/runtime
run=/root/autodl-tmp/experiments/qam_prompt_fixcheck_20260801_v1
date --iso-8601=seconds
printf 'PROCESSES_BEGIN\n'
pgrep -af '[q]am_prompt_fixcheck|[t]rain_embodied_agent.py|[t]orch.distributed.run|[r]aylet' || true
printf 'PROCESSES_END\n'
printf 'EXIT='; test -f "$runtime/exit_code.txt" && cat "$runtime/exit_code.txt" || printf 'running\n'
printf 'DRIVER_TAIL_BEGIN\n'
tail -n 60 "$runtime/driver.log" 2>/dev/null || true
printf 'DRIVER_TAIL_END\n'
printf 'METRIC_KEYS_BEGIN\n'
find "$run" -type f -name metrics.log -print -exec tail -n 8 {} \; 2>/dev/null || true
printf 'METRIC_KEYS_END\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'MEM_CURRENT='; cat /sys/fs/cgroup/memory.current 2>/dev/null || true
printf 'MEM_EVENTS='; tr '\n' ' ' </sys/fs/cgroup/memory.events 2>/dev/null || true; printf '\n'
