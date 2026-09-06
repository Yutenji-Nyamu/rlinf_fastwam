#!/usr/bin/env bash
set -euo pipefail

run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821
source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
uploaded_launcher=/root/autodl-tmp/idea2_dvac_r_only_downweight_run_smoke_20260821.sh
uploaded_observer=/root/autodl-tmp/idea2_dvac_r_only_downweight_observe_resources_20260821.sh

printf 'IDENTITY\n'
hostname
pwd
id -u

printf 'SOURCE\n'
git -C "$source_root" rev-parse HEAD
git -C "$source_root" status --short

printf 'PRELAUNCH_RESOURCES\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.current

test ! -e "$run_dir"
test ! -e "$runtime_dir"
bash -n "$uploaded_launcher"
bash -n "$uploaded_observer"

mkdir -p "$runtime_dir"
mv "$uploaded_launcher" "$runtime_dir/launch_smoke.sh"
mv "$uploaded_observer" "$runtime_dir/observe_resources.sh"
chmod 700 "$runtime_dir/launch_smoke.sh" "$runtime_dir/observe_resources.sh"

nohup setsid bash "$runtime_dir/launch_smoke.sh" \
  > "$runtime_dir/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$runtime_dir/wrapper.pid"

for _ in $(seq 1 20); do
  if test -s "$runtime_dir/driver.pid" && test -s "$runtime_dir/observer.pid"; then
    break
  fi
  sleep 1
done

printf 'LAUNCHED\n'
printf 'wrapper_pid=%s\n' "$wrapper_pid"
printf 'driver_pid=%s\n' "$(cat "$runtime_dir/driver.pid")"
printf 'observer_pid=%s\n' "$(cat "$runtime_dir/observer.pid")"
printf 'started_at=%s\n' "$(cat "$runtime_dir/launch_started_at.txt")"
ps -o pid=,ppid=,stat=,etime=,args= -p \
  "$wrapper_pid,$(cat "$runtime_dir/driver.pid"),$(cat "$runtime_dir/observer.pid")"
sleep 5
printf 'DRIVER_LOG_TAIL\n'
tail -n 20 "$runtime_dir/driver.log"
printf 'RESOURCE_ROWS\n'
wc -l "$runtime_dir/resource_monitor/resources.csv"
