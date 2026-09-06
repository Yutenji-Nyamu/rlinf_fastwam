set -u
runtime=/root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime
run=/root/autodl-tmp/experiments/qam_formal_20260801_v2
date --iso-8601=seconds
printf 'SUPERVISOR='; pgrep -fc '[q]am_formal_launch_20260801_v2.sh' || true
printf 'TRAIN_PIDS='; pgrep -fc '[t]rain_embodied_agent.py' || true
printf 'RAYLETS='; pgrep -xc raylet || true
printf 'EXIT='; test -f "$runtime/exit_code.txt" && cat "$runtime/exit_code.txt" || printf 'running\n'
printf 'PROVENANCE_BEGIN\n'; cat "$runtime/run_provenance.tsv" 2>/dev/null || true; printf 'PROVENANCE_END\n'
printf 'ERRORS_BEGIN\n'
grep -E 'ValueError: QAM|CUDA out of memory|RayActorError|NCCL|PRECHECK_FAIL' "$runtime/driver.log" /root/autodl-tmp/qam_formal_supervisor_20260801_v2.log 2>/dev/null | tail -n 12 || true
printf 'ERRORS_END\n'
printf 'DRIVER_TAIL_BEGIN\n'; tail -n 20 "$runtime/driver.log" 2>/dev/null || true; printf 'DRIVER_TAIL_END\n'
find "$run" -type f -name metrics.log -exec tail -n 3 {} \; 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'CGROUP_CURRENT='; cat /sys/fs/cgroup/memory.current 2>/dev/null || true
printf 'CGROUP_ANON='; awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || true
printf 'OOM_EVENTS='; awk '$1=="oom"||$1=="oom_kill"{printf "%s=%s ",$1,$2}' /sys/fs/cgroup/memory.events 2>/dev/null || true; printf '\n'
printf 'DISK_AVAILABLE='; df -B1 --output=avail /root/autodl-tmp | tail -n 1
