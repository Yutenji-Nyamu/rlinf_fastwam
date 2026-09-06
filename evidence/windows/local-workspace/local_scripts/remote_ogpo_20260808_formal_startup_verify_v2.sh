#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

venv=/root/autodl-tmp/RLinf/.venv
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
driver_pid=$(cat "$runtime_root/driver_pid.txt")
monitor_pid=$(cat "$runtime_root/monitor_pid.txt")

kill -0 "$driver_pid"
kill -0 "$monitor_pid"
test ! -e "$runtime_root/exit_code.txt"
grep -aFq 'Evaluating Rollout Epochs: 100%' "$runtime_root/driver.log"
grep -aFq 'Generating Rollout Epochs:' "$runtime_root/driver.log"

printf 'VERIFIED_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'DRIVER_PID\t%s\n' "$driver_pid"
printf 'MONITOR_PID\t%s\n' "$monitor_pid"
printf 'RESOURCE_ROWS\t%s\n' "$(($(wc -l <"$runtime_root/resources_1s.csv") - 1))"

EVENT_ROOT="$run_root/tensorboard" "${venv}/bin/python" -B - <<'PY'
import os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

events = sorted(Path(os.environ["EVENT_ROOT"]).glob("events.out.tfevents.*"))
assert len(events) == 1, events
acc = EventAccumulator(str(events[0])).Reload()
tags = acc.Tags().get("scalars", [])
print("EVENT_FILE", events[0], sep="\t")
print("SCALAR_TAG_COUNT", len(tags), sep="\t")
for tag in sorted(t for t in tags if t.startswith("eval/")):
    values = acc.Scalars(tag)
    if values:
        item = values[-1]
        print("EVAL_SCALAR", tag, item.step, item.value, sep="\t")
PY

awk -F, '
  NR == 2 {min_host=$2; min_disk=$9; max_cgroup=$3; max_g0=$10; max_g1=$15}
  NR > 1 {
    if ($2 < min_host) min_host=$2;
    if ($9 < min_disk) min_disk=$9;
    if ($3 > max_cgroup) max_cgroup=$3;
    if ($10 > max_g0) max_g0=$10;
    if ($15 > max_g1) max_g1=$15;
    oom=$6; oom_kill=$7
  }
  END {
    printf "RESOURCE_SUMMARY\tmin_host_available=%s\tmax_cgroup=%s\tmin_disk_available=%s\tmax_gpu0_mib=%s\tmax_gpu1_mib=%s\toom=%s\toom_kill=%s\n", min_host, max_cgroup, min_disk, max_g0, max_g1, oom, oom_kill
  }
' "$runtime_root/resources_1s.csv"

printf '%s\n' '=== latest train/eval transition ==='
grep -aE 'Evaluating Rollout Epochs|Generating Rollout Epochs|success_once|eval/' \
  "$runtime_root/driver.log" | tail -n 20 || true
printf '%s\n' OGPO_FORMAL_V2_STARTUP_HEALTH_VERIFIED
