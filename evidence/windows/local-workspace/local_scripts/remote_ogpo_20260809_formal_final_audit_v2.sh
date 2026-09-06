#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
runtime=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
python=/root/autodl-tmp/RLinf/.venv/bin/python
driver_pid=$(cat "$runtime/driver_pid.txt" 2>/dev/null || printf '0')
monitor_pid=$(cat "$runtime/monitor_pid.txt" 2>/dev/null || printf '0')

printf 'AUDIT_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'HOST\t%s\n' "$(hostname)"
printf 'DRIVER_PID\t%s\n' "$driver_pid"
kill -0 "$driver_pid" 2>/dev/null && printf 'DRIVER_ALIVE\t1\n' || printf 'DRIVER_ALIVE\t0\n'
printf 'MONITOR_PID\t%s\n' "$monitor_pid"
kill -0 "$monitor_pid" 2>/dev/null && printf 'MONITOR_ALIVE\t1\n' || printf 'MONITOR_ALIVE\t0\n'
printf 'EXIT_CODE\t%s\n' "$(cat "$runtime/exit_code.txt" 2>/dev/null || printf PENDING)"
printf 'MONITOR_EXIT_CODE\t%s\n' "$(cat "$runtime/monitor_exit_code.txt" 2>/dev/null || printf PENDING)"
printf 'STARTED_AT\t%s\n' "$(cat "$runtime/started_at.txt" 2>/dev/null || printf PENDING)"
printf 'FINISHED_AT\t%s\n' "$(cat "$runtime/finished_at.txt" 2>/dev/null || printf PENDING)"

printf '%s\n' '=== repository ==='
git -C "$repo" rev-parse HEAD
git -C "$repo" branch --show-current
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
timeout 15 git -C "$repo" ls-remote personal refs/heads/codex/ogpo-pi0-robotwin || true

printf '%s\n' '=== immutable runtime hashes ==='
sha256sum "$runtime/source_config.yaml" "$runtime/resolved.yaml" "$runtime/exact_command.txt" \
  "$runtime/run_provenance.tsv" "$runtime/stop_conditions.txt"

printf '%s\n' '=== lightweight artifacts ==='
for path in \
  "$runtime/driver.log" "$runtime/resources_1s.csv" "$runtime/resolved.yaml" \
  "$runtime/source_config.yaml" "$runtime/exact_command.txt" "$runtime/run_provenance.tsv" \
  "$runtime/stop_conditions.txt" "$runtime/exit_code.txt" "$runtime/started_at.txt" \
  "$runtime/finished_at.txt" "$runtime/resources_before.txt" "$runtime/resources_after.txt" \
  "$run_root/metrics.log" "$run_root/tensorboard/config.yaml" \
  "$run_root"/tensorboard/events.out.tfevents.*; do
  test -e "$path" && stat -c '%s\t%Y\t%n' "$path"
done

printf '%s\n' '=== checkpoints ==='
"$python" -B - "$run_root" <<'PY'
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoints = sorted(root.glob("*/checkpoints/global_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[-1]))
out = []
for ckpt in checkpoints:
    files = [p for p in ckpt.rglob("*") if p.is_file()]
    components = {}
    for path in files:
        rel = path.relative_to(ckpt)
        key = str(rel.parts[0]) if rel.parts else "root"
        if len(rel.parts) >= 3 and rel.parts[:2] == ("actor", "ogpo_components"):
            key = "actor/ogpo_components"
        elif len(rel.parts) >= 3 and rel.parts[:2] == ("actor", "dcp_checkpoint"):
            key = "actor/dcp_checkpoint"
        components[key] = components.get(key, 0) + path.stat().st_size
    manifests = list(ckpt.rglob("complete.json"))
    manifest = None
    if manifests:
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    out.append({
        "path": str(ckpt), "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "components": components,
        "complete_manifest_path": str(manifests[0]) if manifests else None,
        "complete": manifest.get("complete") if manifest else None,
        "global_online_rows": manifest.get("global_online_rows") if manifest else None,
        "policy_version": manifest.get("policy_version") if manifest else None,
        "step": manifest.get("step") if manifest else None,
        "contract": manifest.get("contract") if manifest else None,
    })
print(json.dumps(out, indent=2, sort_keys=True))
PY

printf '%s\n' '=== scalar summary ==='
"$python" -B - "$run_root/tensorboard" <<'PY'
import json, math, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS

acc = EventAccumulator(sys.argv[1], size_guidance={SCALARS: 0})
acc.Reload()
out = {}
for tag in sorted(acc.Tags().get("scalars", [])):
    if not (tag.startswith("train/ogpo/") or tag.startswith("env/") or tag.startswith("eval/")):
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
def pct(values, q):
    if not values: return None
    x=sorted(values); pos=(len(x)-1)*q; lo=int(pos); hi=min(lo+1,len(x)-1); f=pos-lo
    return x[lo]*(1-f)+x[hi]*f
keys=("host_available_bytes","cgroup_current_bytes","cgroup_anon_bytes","cgroup_file_bytes",
      "disk_available_bytes","gpu0_used_mib","gpu1_used_mib","gpu0_util_pct","gpu1_util_pct",
      "gpu0_power_w","gpu1_power_w","matched_total_rss_kib","compute_process_count")
out={}
for key in keys:
    x=vals(key)
    out[key]={"min":min(x),"max":max(x),"mean":statistics.fmean(x),"p50":pct(x,.5),"p95":pct(x,.95),"last":x[-1]} if x else None
t=vals("unix_time")
out["meta"]={"rows":len(rows),"first_unix":t[0],"last_unix":t[-1],
             "duration_seconds":t[-1]-t[0],
             "oom_max":max(vals("cgroup_oom_events"), default=None),
             "oom_kill_max":max(vals("cgroup_oom_kill_events"), default=None)}
if len(t) > 1:
    dt=[max(0,b-a) for a,b in zip(t,t[1:])]
    for index in (0,1):
        util=vals(f"gpu{index}_util_pct"); power=vals(f"gpu{index}_power_w")
        n=min(len(dt),len(util)-1,len(power)-1)
        out[f"gpu{index}_util_equivalent_hours"] = sum(dt[i]*util[i]/100 for i in range(n))/3600
        out[f"gpu{index}_energy_kwh"] = sum(dt[i]*power[i] for i in range(n))/3_600_000
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
print(json.dumps({"bytes":len(text.encode('utf-8')),"lines":text.count('\n')+1,
                  "counts":{k:len(re.findall(v,text)) for k,v in patterns.items()}}, sort_keys=True))
PY

printf '%s\n' '=== progress tail ==='
grep -aE 'Global Step:|Elapsed:|Evaluating Rollout Epochs|Generating Rollout Epochs|total_online_rows=|actor_updates=|Saving checkpoint' "$runtime/driver.log" | tail -n 100 || true
printf '%s\n' '=== driver tail ==='
tail -n 80 "$runtime/driver.log"
printf '%s\n' '=== live resources ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits
printf 'CGROUP_CURRENT_BYTES\t'; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
df -B1 /root/autodl-tmp | tail -n 1
