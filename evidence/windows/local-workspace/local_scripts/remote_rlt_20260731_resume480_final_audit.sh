#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1
experiment=robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1
exp_root="$run_root/$experiment"
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1/runtime
python=/root/autodl-tmp/RLinf/.venv/bin/python

echo "SECTION=IDENTITY"
date --iso-8601=seconds
hostname
id -u

echo "SECTION=GIT_AND_CONFIG"
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD' 2>/dev/null || true
for name in resolved.yaml run_provenance.tsv budget.json stop_conditions.txt exact_command.txt; do
  path="$runtime/$name"
  if [[ -f "$path" ]]; then
    stat -c '%n size=%s mtime=%y' "$path"
    sha256sum "$path"
  else
    echo "$path MISSING"
  fi
done

echo "SECTION=PROCESS_AND_RUNTIME"
for name in started_at.txt finished_at.txt exit_code.txt driver_pid.txt monitor_pid.txt; do
  path="$runtime/$name"
  if [[ -f "$path" ]]; then
    printf '%s=' "$name"
    tr '\n' ' ' <"$path"
    echo
  else
    echo "$name=MISSING"
  fi
done
for name in driver monitor; do
  pid_file="$runtime/${name}_pid.txt"
  pid="$(tr -d '[:space:]' <"$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "${name}_pid=$pid alive=1"
    ps -p "$pid" -o pid=,ppid=,lstart=,etimes=,stat=,%cpu=,%mem=,rss=,cmd=
  else
    echo "${name}_pid=${pid:-missing} alive=0"
  fi
done
ps -eo pid=,comm=,args= |
  awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}' || true
pgrep -ax raylet || true
pgrep -ax gcs_server || true

echo "SECTION=LOG"
stat -c '%n size=%s mtime=%y' "$runtime/driver.log"
wc -l -c "$runtime/driver.log"
printf 'global_step_480_count='
grep -c 'Global Step 480/480' "$runtime/driver.log" || true
printf 'checkpoint_480_count='
grep -c 'global_step_480' "$runtime/driver.log" || true
for pattern_name in cuda_oom nccl_fatal actor_death traceback nan_metric; do
  case "$pattern_name" in
    cuda_oom) pattern='CUDA out of memory|CUDA error: out of memory' ;;
    nccl_fatal) pattern='NCCL.*(error|fatal|abort)' ;;
    actor_death) pattern='RayActorError|actor died|worker died' ;;
    traceback) pattern='Traceback \(most recent call last\)' ;;
    nan_metric) pattern='(^|[^[:alpha:]])nan([^[:alpha:]]|$)' ;;
  esac
  printf '%s_count=' "$pattern_name"
  if [[ "$pattern_name" == "nan_metric" ]]; then
    grep -Ei "$pattern" "$runtime/driver.log" |
      grep -Eiv 'curobo|vulkan|warning|traceback' |
      wc -l || true
  else
    grep -Eic "$pattern" "$runtime/driver.log" || true
  fi
done
echo "FATAL_SAMPLES_BEGIN"
grep -Ein 'CUDA out of memory|NCCL.*(error|fatal|abort)|RayActorError|actor died|worker died|Traceback \(most recent call last\)' \
  "$runtime/driver.log" | head -n 30 || true
echo "FATAL_SAMPLES_END"
echo "FINAL_PROGRESS_BEGIN"
grep -E 'Global Step (475|476|477|478|479|480)/480|Saving checkpoint|eval/num_trajectories|eval/success_once|eval/success_at_end|train/success_once|rlt/update_step|rlt/actor_loss|rlt/critic_loss' \
  "$runtime/driver.log" | tail -n 180 || true
echo "FINAL_PROGRESS_END"

echo "SECTION=RESOURCE_CSV"
stat -c '%n size=%s mtime=%y' "$runtime/resources.csv"
"$python" -B - "$runtime/resources.csv" <<'PY'
import csv
import json
import math
import statistics
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

summary = {
    "rows": len(rows),
    "columns": list(rows[0]) if rows else [],
    "first": rows[0] if rows else {},
    "last": rows[-1] if rows else {},
    "numeric": {},
}
if rows:
    for column in rows[0]:
        values = []
        for row in rows:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if not values:
            continue
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        summary["numeric"][column] = {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "p95": ordered[p95_index],
            "last": values[-1],
        }
print(json.dumps(summary, sort_keys=True))
PY

echo "SECTION=TENSORBOARD"
find "$run_root/tensorboard" -maxdepth 1 -type f \
  -printf '%p size=%s mtime=%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort
"$python" -B - "$run_root/tensorboard" <<'PY'
import glob
import json
import math
import os
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = sys.argv[1]
events = sorted(glob.glob(os.path.join(root, "events.out.tfevents.*")))
payload = {"event_files": events, "tags": [], "stats": {}, "eval_points": {}}
if events:
    accumulator = EventAccumulator(events[-1], size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = sorted(accumulator.Tags().get("scalars", []))
    payload["tags"] = tags
    for tag in tags:
        points = [
            [int(point.step), float(point.value), float(point.wall_time)]
            for point in accumulator.Scalars(tag)
            if math.isfinite(float(point.value))
        ]
        values = [point[1] for point in points]
        if values:
            payload["stats"][tag] = {
                "count": len(values),
                "first_step": points[0][0],
                "last_step": points[-1][0],
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
                "first10_mean": statistics.fmean(values[:10]),
                "last10_mean": statistics.fmean(values[-10:]),
            }
        if "eval/" in tag and any(
            token in tag
            for token in ("success_once", "success_at_end", "num_trajectories", "return")
        ):
            payload["eval_points"][tag] = points
print(json.dumps(payload, sort_keys=True))
PY

echo "SECTION=CHECKPOINTS"
checkpoint_root="$exp_root/checkpoints"
total_bytes=0
total_files=0
checkpoint_count=0
for checkpoint in "$checkpoint_root"/global_step_*; do
  [[ -d "$checkpoint" ]] || continue
  step=${checkpoint##*_}
  bytes=$(du -sb "$checkpoint" | awk '{print $1}')
  files=$(find "$checkpoint" -type f | wc -l)
  total_bytes=$((total_bytes + bytes))
  total_files=$((total_files + files))
  checkpoint_count=$((checkpoint_count + 1))
  printf 'checkpoint=%s bytes=%s files=%s\n' "$step" "$bytes" "$files"
  complete="$checkpoint/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"
  if [[ -f "$complete" ]]; then
    printf 'completion_%s=' "$step"
    tr -d '\n' <"$complete"
    echo
    sha256sum "$complete"
  else
    echo "completion_${step}=MISSING"
  fi
  for rank in 0 1; do
    state="$checkpoint/actor/sac_components/rlt_trainer_state/checkpoint_rank_${rank}.pt"
    metadata="$checkpoint/actor/sac_components/replay_buffer/rank_${rank}/metadata.json"
    [[ -f "$state" ]] && sha256sum "$state" || echo "state_${step}_rank${rank}=MISSING"
    if [[ -f "$metadata" ]]; then
      printf 'replay_%s_rank%s=' "$step" "$rank"
      tr -d '\n' <"$metadata"
      echo
    else
      echo "replay_${step}_rank${rank}=MISSING"
    fi
  done
done
printf 'checkpoint_count=%s total_bytes=%s total_files=%s\n' \
  "$checkpoint_count" "$total_bytes" "$total_files"
du -sb "$run_root"
find "$run_root" -type f | wc -l

echo "SECTION=LIVE_RESOURCE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
df -B1 /root/autodl-tmp
printf 'cgroup_current='
cat /sys/fs/cgroup/memory.current
grep -E '^(anon|file|inactive_file|active_file|slab) ' /sys/fs/cgroup/memory.stat
tr '\n' ' ' </sys/fs/cgroup/memory.events
echo
tr '\n' ' ' </proc/pressure/memory
echo

echo "RLT_RESUME480_FINAL_AUDIT_DONE"
