#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
runtime=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1
python=/root/autodl-tmp/RLinf/.venv/bin/python
driver_pid=$(cat "$runtime/driver_pid.txt" 2>/dev/null || printf '0')
monitor_pid=$(cat "$runtime/monitor_pid.txt" 2>/dev/null || printf '0')

printf 'AUDIT_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'HOST\t%s\n' "$(hostname)"
printf 'DRIVER_PID\t%s\n' "$driver_pid"
kill -0 "$driver_pid" 2>/dev/null && printf 'DRIVER_ALIVE\t1\n' || printf 'DRIVER_ALIVE\t0\n'
printf 'MONITOR_PID\t%s\n' "$monitor_pid"
kill -0 "$monitor_pid" 2>/dev/null && printf 'MONITOR_ALIVE\t1\n' || printf 'MONITOR_ALIVE\t0\n'
printf 'EXIT_CODE\t%s\n' "$(cat "$runtime/exit_code.txt" 2>/dev/null || printf 'PENDING')"
printf 'MONITOR_EXIT_CODE\t%s\n' "$(cat "$runtime/monitor_exit_code.txt" 2>/dev/null || printf 'PENDING')"
printf 'STARTED_AT\t%s\n' "$(cat "$runtime/started_at.txt" 2>/dev/null || printf 'PENDING')"
printf 'FINISHED_AT\t%s\n' "$(cat "$runtime/finished_at.txt" 2>/dev/null || printf 'PENDING')"

printf '%s\n' '=== repository ==='
git -C "$repo" rev-parse HEAD
git -C "$repo" branch --show-current
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'

printf '%s\n' '=== immutable runtime hashes ==='
sha256sum "$runtime/source_config.yaml" "$runtime/resolved.yaml" "$runtime/exact_command.txt" \
  "$runtime/run_provenance.tsv" "$runtime/stop_conditions.txt"

printf '%s\n' '=== runtime inventory ==='
find "$runtime" -maxdepth 1 -type f -printf '%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' | sort
printf 'RUNTIME_BYTES\t%s\n' "$(du -sb "$runtime" | awk '{print $1}')"

printf '%s\n' '=== run-root summary ==='
printf 'RUN_ROOT_BYTES\t%s\n' "$(du -sb "$run_root" 2>/dev/null | awk '{print $1}')"
printf 'RUN_FILE_COUNT\t%s\n' "$(find "$run_root" -type f 2>/dev/null | wc -l)"
find "$run_root" -maxdepth 3 -mindepth 1 -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null | sort -k4
printf '%s\n' '=== run files largest ==='
find "$run_root" -type f -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null | sort -nr | head -n 80

printf '%s\n' '=== checkpoint inventory ==='
find "$run_root" -type f \( -path '*/checkpoints/*' -o -name 'manifest.json' -o -name '*.complete' -o -name '*sidecar*' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null | sort -k3 || true

printf '%s\n' '=== resource summary ==='
"$python" -B - "$runtime/resources_1s.csv" <<'PY'
import csv
import json
import math
import statistics
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

def nums(key):
    out = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            out.append(value)
    return out

def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

keys = [
    "host_available_bytes", "cgroup_current_bytes", "cgroup_anon_bytes", "cgroup_file_bytes",
    "shm_used_bytes", "disk_available_bytes", "gpu0_used_mib", "gpu1_used_mib",
    "gpu0_util_pct", "gpu1_util_pct", "gpu0_power_w", "gpu1_power_w",
    "env_rss_kib", "actor_rss_kib", "rollout_rss_kib", "driver_rss_kib",
    "ray_rss_kib", "matched_total_rss_kib", "compute_process_count",
]
summary = {}
for key in keys:
    values = nums(key)
    summary[key] = {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "last": values[-1] if values else None,
    }
times = nums("unix_time")
gaps = [b - a for a, b in zip(times, times[1:]) if b >= a]
summary["meta"] = {
    "rows": len(rows),
    "first_unix": times[0] if times else None,
    "last_unix": times[-1] if times else None,
    "duration_seconds": times[-1] - times[0] if len(times) >= 2 else 0,
    "cadence_p50_seconds": percentile(gaps, 0.50),
    "cadence_p95_seconds": percentile(gaps, 0.95),
    "oom_max": max(nums("cgroup_oom_events"), default=None),
    "oom_kill_max": max(nums("cgroup_oom_kill_events"), default=None),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY

printf '%s\n' '=== scalar latest ==='
"$python" -B - "$run_root/tensorboard" <<'PY'
import json
import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS

acc = EventAccumulator(sys.argv[1], size_guidance={SCALARS: 0})
acc.Reload()
result = {}
for tag in sorted(acc.Tags().get("scalars", [])):
    if not (tag.startswith("train/ogpo/") or tag.startswith("eval/")):
        continue
    values = acc.Scalars(tag)
    if not values:
        continue
    last = values[-1]
    result[tag] = {
        "count": len(values),
        "first_step": values[0].step,
        "first_value": values[0].value if math.isfinite(values[0].value) else str(values[0].value),
        "last_step": last.step,
        "last_value": last.value if math.isfinite(last.value) else str(last.value),
        "last_wall_time": last.wall_time,
        "min": min(v.value for v in values),
        "max": max(v.value for v in values),
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY

printf '%s\n' '=== driver diagnostics ==='
"$python" -B - "$runtime/driver.log" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
patterns = {
    "traceback": r"Traceback \(most recent call last\):",
    "cuda_oom": r"CUDA out of memory|OutOfMemoryError",
    "actor_died": r"ActorDiedError",
    "ray_task_error": r"RayTaskError",
    "nan_token": r"(?i)(?<![A-Za-z])nan(?![A-Za-z])",
    "inf_token": r"(?i)(?<![A-Za-z])(?:\+|-)?inf(?:inity)?(?![A-Za-z])",
    "checkpoint_save": r"Saving checkpoint at step",
    "evaluation_start": r"Evaluating Rollout Epochs",
    "training_rollout_complete": r"Generating Rollout Epochs:\s+100%",
}
print(json.dumps({
    "bytes": path.stat().st_size,
    "lines": text.count("\n") + 1,
    "counts": {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()},
}, indent=2, sort_keys=True))
PY
printf '%s\n' '=== traceback contexts ==='
grep -an -B 4 -A 16 'Traceback (most recent call last):' "$runtime/driver.log" | head -n 160 || true
printf '%s\n' '=== driver progress tail ==='
grep -aE 'Generating Rollout Epochs|Evaluating Rollout Epochs|online_rows|total_online_rows|actor_updates|critic_updates|Saving checkpoint' \
  "$runtime/driver.log" | tail -n 120 || true
printf '%s\n' '=== driver tail ==='
tail -n 120 "$runtime/driver.log"

printf '%s\n' '=== live processes ==='
ps -eo pid,ppid,stat,lstart,etimes,rss,pcpu,cmd --sort=pid \
  | grep -E 'train_embodied_agent|raylet|gcs_server|EnvWorker|ActorGroup|RolloutGroup' \
  | grep -v -E 'grep -E|formal_artifact_audit_v1' || true
printf '%s\n' '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format=csv,noheader,nounits
printf '%s\n' '=== compute apps ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
printf '%s\n' '=== memory and disk ==='
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
printf 'CGROUP_CURRENT_BYTES\t'; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -B1 /root/autodl-tmp /dev/shm
