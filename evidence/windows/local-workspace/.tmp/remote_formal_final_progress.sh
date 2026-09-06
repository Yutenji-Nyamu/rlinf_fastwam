set -euo pipefail

RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
date '+TIME=%Y-%m-%d %H:%M:%S %Z'
kill -0 70062
echo "DRIVER_ALIVE=1"
grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1
grep 'sac/global_resident_transitions=' "$RUN/formal_driver.log" | tail -n 1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
grep -E '^(anon|file|inactive_file) ' /sys/fs/cgroup/memory.stat
grep -E '^(oom|oom_kill) ' /sys/fs/cgroup/memory.events
