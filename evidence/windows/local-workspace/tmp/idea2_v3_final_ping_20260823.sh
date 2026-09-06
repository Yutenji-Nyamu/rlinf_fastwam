set -u
RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
date '+%Y-%m-%d %H:%M:%S %Z'
for role in wrapper driver observer; do
  pid=$(cat "$RUNTIME/$role.pid")
  ps -p "$pid" -o pid=,stat=,etime=,cmd= || true
done
grep -a 'Global Step:' "$RUN/metrics.log" | tail -n 1
grep -a 'Generating Rollout Epochs:' "$RUNTIME/driver.log" | tail -n 2 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.events='; tr '\n' ' ' < /sys/fs/cgroup/memory.events; echo
