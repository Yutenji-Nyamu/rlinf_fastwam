#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1

runtime=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
python=/root/autodl-tmp/RLinf/.venv/bin/python
driver_pid=$(cat "$runtime/driver_pid.txt" 2>/dev/null || printf '0')
monitor_pid=$(cat "$runtime/monitor_pid.txt" 2>/dev/null || printf '0')

printf 'AUDIT_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'DRIVER_ALIVE\t'; kill -0 "$driver_pid" 2>/dev/null && printf '1\n' || printf '0\n'
printf 'MONITOR_ALIVE\t'; kill -0 "$monitor_pid" 2>/dev/null && printf '1\n' || printf '0\n'
printf 'EXIT_CODE\t%s\n' "$(cat "$runtime/exit_code.txt" 2>/dev/null || printf PENDING)"
printf 'STARTED_AT\t%s\n' "$(cat "$runtime/started_at.txt")"
printf 'FINISHED_AT\t%s\n' "$(cat "$runtime/finished_at.txt" 2>/dev/null || printf PENDING)"

printf '%s\n' '=== light artifacts ==='
for path in \
  "$runtime/driver.log" "$runtime/resources_1s.csv" "$runtime/resolved.yaml" \
  "$runtime/source_config.yaml" "$runtime/exact_command.txt" "$runtime/run_provenance.tsv" \
  "$runtime/stop_conditions.txt" "$run_root/metrics.log" "$run_root/tensorboard/config.yaml" \
  "$run_root"/tensorboard/events.out.tfevents.*; do
  test -e "$path" && stat -c '%s\t%Y\t%n' "$path"
done

printf '%s\n' '=== checkpoints ==='
find "$run_root/checkpoints" -maxdepth 2 -mindepth 1 \
  -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null | sort || true
printf 'CHECKPOINT_BYTES\t%s\n' "$(du -sb "$run_root/checkpoints" 2>/dev/null | awk '{print $1}' || printf 0)"

printf '%s\n' '=== scalar summary ==='
"$python" -B - "$run_root/tensorboard" <<'PY'
import json, math, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS

acc = EventAccumulator(sys.argv[1], size_guidance={SCALARS: 0})
acc.Reload()
wanted = (
    "train/ogpo/", "train/success_once", "train/return", "train/episode_len",
    "eval/success_once", "eval/return", "eval/episode_len",
)
out = {}
for tag in sorted(acc.Tags().get("scalars", [])):
    if not any(tag.startswith(prefix) for prefix in wanted):
        continue
    values = acc.Scalars(tag)
    if not values:
        continue
    finite = [v.value for v in values if math.isfinite(v.value)]
    out[tag] = {
        "count": len(values), "first_step": values[0].step,
        "first": values[0].value, "last_step": values[-1].step,
        "last": values[-1].value,
        "min": min(finite) if finite else None, "max": max(finite) if finite else None,
    }
print(json.dumps(out, sort_keys=True, indent=2))
PY

printf '%s\n' '=== resource summary ==='
"$python" -B - "$runtime/resources_1s.csv" <<'PY'
import csv, json, math, statistics, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
def vals(key):
    out=[]
    for row in rows:
        try: value=float(row[key])
        except (KeyError, TypeError, ValueError): continue
        if math.isfinite(value) and value >= 0: out.append(value)
    return out
keys=("host_available_bytes","cgroup_current_bytes","disk_available_bytes",
      "gpu0_used_mib","gpu1_used_mib","gpu0_util_pct","gpu1_util_pct",
      "gpu0_power_w","gpu1_power_w","matched_total_rss_kib")
out={}
for key in keys:
    x=vals(key)
    out[key]={"min":min(x),"max":max(x),"mean":statistics.fmean(x),"last":x[-1]} if x else None
t=vals("unix_time")
out["meta"]={"rows":len(rows),"first_unix":t[0],"last_unix":t[-1],
             "duration_seconds":t[-1]-t[0],
             "oom_max":max(vals("cgroup_oom_events"), default=None),
             "oom_kill_max":max(vals("cgroup_oom_kill_events"), default=None)}
print(json.dumps(out, sort_keys=True, indent=2))
PY

printf '%s\n' '=== driver diagnostics ==='
"$python" -B - "$runtime/driver.log" <<'PY'
import json, re, sys
text=open(sys.argv[1], encoding="utf-8", errors="replace").read()
patterns={
  "traceback":r"Traceback \(most recent call last\):",
  "cuda_oom":r"CUDA out of memory|OutOfMemoryError",
  "actor_died":r"ActorDiedError", "ray_task_error":r"RayTaskError",
  "nan":r"(?i)(?<![A-Za-z])nan(?![A-Za-z])",
  "checkpoint_save":r"Saving checkpoint at step",
  "eval_complete":r"Evaluating Rollout Epochs:\s+100%",
  "train_complete":r"Generating Rollout Epochs:\s+100%",
}
print(json.dumps({k:len(re.findall(v,text)) for k,v in patterns.items()}, sort_keys=True))
PY

printf '%s\n' '=== progress tail ==='
grep -aE 'Global Step:|Elapsed:|Evaluating Rollout Epochs|Generating Rollout Epochs|total_online_rows=|actor_updates=|Saving checkpoint' "$runtime/driver.log" | tail -n 70 || true
printf '%s\n' '=== live resources ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits
printf 'CGROUP_CURRENT_BYTES\t'; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
grep -E 'MemTotal|MemAvailable' /proc/meminfo
df -B1 /root/autodl-tmp | tail -n 1
