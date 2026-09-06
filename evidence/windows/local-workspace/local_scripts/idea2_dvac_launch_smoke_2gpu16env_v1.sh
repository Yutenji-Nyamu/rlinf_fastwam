set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
robotwin=/root/autodl-tmp/RoboTwin_RLinf
runtime=/root/autodl-tmp/RLinf/.venv/bin/python
run_id=idea2_dvac_sft_smoke_2gpu_16env_v1
config=/root/autodl-tmp/idea2_dvac_run_configs/$run_id.yaml
output="$target/outputs/$run_id"
runtime_dir=/root/autodl-tmp/idea2_dvac_runtime/$run_id
observer="$runtime_dir/observe_resources.sh"
monitor_dir="$runtime_dir/resource_monitor"
expected_head=61996e15cc7f5a32bd6012b61b20893d94636c82
expected_config_sha=d4b7393e1a02f118f6ea5ab2d303f92e83779d3e2a968a7920aebadbd35bc018

test "$(git -C "$target" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$target" rev-parse personal/codex/idea2-dvac-pi0-robotwin)" = "$expected_head"
test -z "$(git -C "$target" status --porcelain)"
test "$(sha256sum "$config" | cut -d' ' -f1)" = "$expected_config_sha"
test -x "$runtime"
test -d "$robotwin"
test -f "$observer"
test ! -e "$output"

mkdir -p "$monitor_dir"
printf 'LAUNCHER_STARTED_AT=%s\n' "$(date --iso-8601=seconds)" > "$runtime_dir/status.env"
{
  date --iso-8601=seconds
  nvidia-smi
  printf 'memory.current='; cat /sys/fs/cgroup/memory.current
  printf 'memory.events='; tr '\n' ';' < /sys/fs/cgroup/memory.events; printf '\n'
  free -h
  df -h /root/autodl-tmp /dev/shm
} > "$monitor_dir/resource_before.txt" 2>&1

cd "$target"
CUDA_VISIBLE_DEVICES=0,1 \
REPO_PATH="$target" \
EMBODIED_PATH="$target/examples/embodiment" \
PYTHONPATH="$target:$robotwin" \
  "$runtime" evaluations/eval_embodied_agent.py \
    --config-path /root/autodl-tmp/idea2_dvac_run_configs \
    --config-name "$run_id" \
    > "$runtime_dir/driver.log" 2>&1 &
driver_pid=$!
printf 'DRIVER_PID=%s\n' "$driver_pid" >> "$runtime_dir/status.env"

bash "$observer" "$driver_pid" "$monitor_dir" \
  > "$monitor_dir/observer.log" 2>&1 &
observer_pid=$!
printf 'OBSERVER_PID=%s\nSTATUS=RUNNING\n' "$observer_pid" >> "$runtime_dir/status.env"

set +e
wait "$driver_pid"
driver_rc=$?
set -e
wait "$observer_pid" || true

{
  date --iso-8601=seconds
  nvidia-smi
  printf 'memory.current='; cat /sys/fs/cgroup/memory.current
  printf 'memory.events='; tr '\n' ';' < /sys/fs/cgroup/memory.events; printf '\n'
  free -h
  df -h /root/autodl-tmp /dev/shm
} > "$monitor_dir/resource_after.txt" 2>&1

printf 'DRIVER_EXIT_CODE=%s\nLAUNCHER_FINISHED_AT=%s\nSTATUS=%s\n' \
  "$driver_rc" "$(date --iso-8601=seconds)" \
  "$(if test "$driver_rc" -eq 0; then printf COMPLETE; else printf FAILED; fi)" \
  >> "$runtime_dir/status.env"
printf '%s\n' "$driver_rc" > "$runtime_dir/driver_exit_code.txt"
exit "$driver_rc"
