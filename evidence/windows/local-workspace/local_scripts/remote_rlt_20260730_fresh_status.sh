#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime
run=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
driver_pid="$(cat "${runtime}/driver_pid.txt")"
monitor_pid="$(cat "${runtime}/monitor_pid.txt")"

printf 'status_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'driver_pid\t%s\n' "${driver_pid}"
printf 'driver_alive\t%s\n' "$(
  kill -0 "${driver_pid}" 2>/dev/null && printf yes || printf no
)"
printf 'monitor_pid\t%s\n' "${monitor_pid}"
printf 'monitor_alive\t%s\n' "$(
  kill -0 "${monitor_pid}" 2>/dev/null && printf yes || printf no
)"
for name in started_at.txt finished_at.txt exit_code.txt; do
  printf '%s\t' "${name}"
  if test -f "${runtime}/${name}"; then
    tr '\n' ' ' <"${runtime}/${name}"
    printf '\n'
  else
    printf '%s\n' MISSING
  fi
done
printf 'driver_log_bytes\t%s\n' "$(stat -c %s "${runtime}/driver.log")"
printf 'driver_log_lines\t%s\n' "$(wc -l <"${runtime}/driver.log")"
printf 'run_bytes\t%s\n' "$(
  test -e "${run}" && du -sb "${run}" | cut -f1 || printf 0
)"
printf 'checkpoints_begin\n'
find "${run}" -type d -path '*/checkpoints/global_step_*' -print 2>/dev/null \
  | LC_ALL=C sort || true
printf 'checkpoints_end\n'

RUNTIME="${runtime}" /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ["RUNTIME"]) / "resources.csv"
rows = list(csv.DictReader(path.open()))
print(f"resource_samples\t{len(rows)}")
if rows:
    numeric = {
        key: [float(row[key]) for row in rows]
        for key in rows[0]
        if key != "unix_time"
    }
    print(f"elapsed_seconds\t{int(float(rows[-1]['unix_time']) - float(rows[0]['unix_time']))}")
    print(
        "gpu_peak_mib\t"
        f"{max(numeric['gpu0_used_mib']):.0f},"
        f"{max(numeric['gpu1_used_mib']):.0f}"
    )
    print(
        "gpu_current_mib\t"
        f"{numeric['gpu0_used_mib'][-1]:.0f},"
        f"{numeric['gpu1_used_mib'][-1]:.0f}"
    )
    print(
        "gpu_util_peak_pct\t"
        f"{max(numeric['gpu0_util_pct']):.0f},"
        f"{max(numeric['gpu1_util_pct']):.0f}"
    )
    print(f"host_available_min_bytes\t{min(numeric['host_available_bytes']):.0f}")
    print(f"cgroup_anon_peak_bytes\t{max(numeric['cgroup_anon_bytes']):.0f}")
    print(f"cgroup_file_peak_bytes\t{max(numeric['cgroup_file_bytes']):.0f}")
    print(f"matched_rss_peak_kib\t{max(numeric['matched_total_rss_kib']):.0f}")
    print(f"disk_available_min_bytes\t{min(numeric['disk_available_bytes']):.0f}")
    for key in ("cgroup_oom_events", "cgroup_oom_kill_events"):
        print(f"{key}_delta\t{numeric[key][-1] - numeric[key][0]:.0f}")
PY

printf 'error_scan_begin\n'
grep -Ein \
  'traceback|cuda out of memory|outofmemory|nccl.*error|nan|inf|rank.*(died|death)|segmentation|fatal|error:' \
  "${runtime}/driver.log" \
  | tail -n 30 || printf '%s\n' NONE
printf 'error_scan_end\n'
printf 'driver_tail_begin\n'
tail -n 100 "${runtime}/driver.log"
printf 'driver_tail_end\n'
