#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
cd "/root/autodl-tmp/RLinf_rlt_pi0_robotwin"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test -z "$(git status --short)"
test "$(sha256sum "examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml" | cut -d' ' -f1)" = "f089f333839c99b87d546e8bcf0d5bddbb7da380e8cc1597e1de4c4450592850"
test "$(sha256sum "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" | cut -d' ' -f1)" = "71cccde9b7f18ab63a10817f75b7d5a4d5f5c8d9cadfef99da20690d327c4766"
test "$(sha256sum "toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py" | cut -d' ' -f1)" = "3278a8cbdf766d30309856eac2a4eb5f8cc3c792986e230c2ef022b615553bb6"
test "$(sha256sum "/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh" | cut -d' ' -f1)" = "925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b"
test "$(sha256sum "/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/resolved.yaml" | cut -d' ' -f1)" = "efff00b71d8ab618f4a77c082cbec8fd65fda9abe2573def31e0aca980e50178"
test ! -e "/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1"
mapfile -t active_rows < <(
  pgrep -af 'train_embodied_agent.py|raylet|gcs_server' \
    || true
)
test "${#active_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF'
)
test "${#compute_rows[@]}" = 0
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 419430400
test "$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)" -ge 214748364800
nohup "/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/run_foreground.sh" \
  >"/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/driver.log" \
  2>&1 \
  </dev/null &
driver_pid=$!
printf '%s\n' "${driver_pid}" >"/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/driver_pid.txt"
nohup bash "/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh" \
  "${driver_pid}" \
  "/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/resources.csv" \
  2 \
  >"/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/monitor.log" \
  2>&1 \
  </dev/null &
monitor_pid=$!
printf '%s\n' "${monitor_pid}" >"/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/monitor_pid.txt"
sleep 2
kill -0 "${driver_pid}"
printf 'FORMAL_DRIVER_PID\t%s\n' "${driver_pid}"
printf 'FORMAL_MONITOR_PID\t%s\n' "${monitor_pid}"
printf 'RUNTIME_ROOT\t%s\n' "/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime"
printf 'RUN_ROOT\t%s\n' "/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1"
printf 'EXPERIMENT_NAME\t%s\n' "robotwin_adjust_bottle_rlt_stage2_formal_100c_v1"
printf 'RESOLVED_SHA256\t%s\n' "efff00b71d8ab618f4a77c082cbec8fd65fda9abe2573def31e0aca980e50178"
