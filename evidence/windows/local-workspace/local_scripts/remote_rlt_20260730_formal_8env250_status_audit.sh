#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1
experiment=robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1
exp_root="$run_root/$experiment"
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime
python=/root/autodl-tmp/RLinf/.venv/bin/python

echo "SECTION=IDENTITY"
date --iso-8601=seconds
hostname
pwd
id -u

echo "SECTION=GIT"
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD' 2>/dev/null || true

echo "SECTION=PROCESS"
for name in driver monitor; do
  pid_file="$runtime/${name}_pid.txt"
  if [[ -f "$pid_file" ]]; then
    pid=$(tr -d '[:space:]' < "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "${name}_pid=$pid alive=1"
      ps -p "$pid" -o pid=,ppid=,lstart=,etimes=,stat=,%cpu=,%mem=,rss=,cmd=
    else
      echo "${name}_pid=$pid alive=0"
    fi
  else
    echo "${name}_pid_file=MISSING"
  fi
done
mapfile -t train_pids < <(
  ps -eo pid=,comm=,args= |
    awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print $1}'
)
printf 'train_pid_count=%s\n' "${#train_pids[@]}"
if ((${#train_pids[@]})); then
  ps -o pid=,ppid=,lstart=,etimes=,stat=,%cpu=,%mem=,rss=,cmd= \
    -p "$(IFS=,; echo "${train_pids[*]}")" || true
fi
for name in raylet gcs_server; do
  pgrep -ax "$name" || true
done

echo "SECTION=RUNTIME_FILES"
for name in started_at.txt finished_at.txt exit_code.txt driver_pid.txt monitor_pid.txt \
  driver.log resources.csv exact_command.txt resolved.yaml run_provenance.tsv budget.json \
  stop_conditions.txt; do
  path="$runtime/$name"
  if [[ -e "$path" ]]; then
    stat -c '%n size=%s mtime=%y' "$path"
    case "$name" in
      started_at.txt|finished_at.txt|exit_code.txt|driver_pid.txt|monitor_pid.txt)
        printf '%s=' "$name"
        tr '\n' ' ' < "$path"
        echo
        ;;
      resolved.yaml|run_provenance.tsv|budget.json|stop_conditions.txt)
        sha256sum "$path"
        ;;
    esac
  else
    echo "$path MISSING"
  fi
done

echo "SECTION=LOG_SUMMARY"
if [[ -f "$runtime/driver.log" ]]; then
  wc -l -c "$runtime/driver.log"
  printf 'epoch_start_count='
  grep -c 'Generating Rollout Epochs' "$runtime/driver.log" || true
  printf 'fatal_cuda_oom_count='
  grep -Eic 'CUDA out of memory|CUDA error: out of memory' "$runtime/driver.log" || true
  printf 'fatal_nccl_count='
  grep -Eic 'NCCL.*(error|fatal|abort)' "$runtime/driver.log" || true
  printf 'ray_actor_death_count='
  grep -Eic 'RayActorError|actor died|worker died' "$runtime/driver.log" || true
  printf 'nan_metric_line_count='
  grep -Ei '(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$runtime/driver.log" |
    grep -Eiv 'curobo|vulkan|warning|traceback' |
    wc -l || true
  printf 'traceback_count='
  grep -c 'Traceback (most recent call last)' "$runtime/driver.log" || true
  printf 'eval_denominator_20_count='
  grep -Ec 'eval/num_trajectories[^0-9]*20([^0-9]|$)' "$runtime/driver.log" || true
  echo "LOG_TAIL_BEGIN"
  tail -n 160 "$runtime/driver.log"
  echo "LOG_TAIL_END"
fi

echo "SECTION=RESOURCE_CSV"
if [[ -f "$runtime/resources.csv" ]]; then
  "$python" -B - "$runtime/resources.csv" <<'PY'
import csv
import json
import math
import statistics
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

def numeric(values):
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result

summary = {
    "rows": len(rows),
    "columns": list(rows[0]) if rows else [],
    "first": rows[0] if rows else {},
    "last": rows[-1] if rows else {},
    "numeric": {},
}
if rows:
    for column in rows[0]:
        values = numeric(row.get(column) for row in rows)
        if not values:
            continue
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        summary["numeric"][column] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "p95": ordered[p95_index],
            "last": values[-1],
        }
print("RESOURCE_JSON_BEGIN")
print(json.dumps(summary, sort_keys=True))
print("RESOURCE_JSON_END")
PY
fi

echo "SECTION=LIVE_RESOURCE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
df -B1 /root/autodl-tmp
for path in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.stat \
  /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.pressure; do
  if [[ -f "$path" ]]; then
    echo "FILE=$path"
    case "$path" in
      */memory.stat) grep -E '^(anon|file|inactive_file|active_file|slab) ' "$path" ;;
      *) cat "$path" ;;
    esac
  fi
done

echo "SECTION=TENSORBOARD"
if [[ -d "$run_root/tensorboard" ]]; then
  find "$run_root/tensorboard" -maxdepth 2 -type f \
    -printf '%p size=%s mtime=%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort
  PYTHONDONTWRITEBYTECODE=1 "$python" -B - "$run_root/tensorboard" <<'PY'
import glob
import json
import math
import os
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = sys.argv[1]
events = sorted(glob.glob(os.path.join(root, "events.out.tfevents.*")))
payload = {"event_files": events, "tags": [], "series": {}, "stats": {}}
if events:
    accumulator = EventAccumulator(events[-1], size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = sorted(accumulator.Tags().get("scalars", []))
    payload["tags"] = tags
    for tag in tags:
        points = accumulator.Scalars(tag)
        series = [
            [int(point.step), float(point.value), float(point.wall_time)]
            for point in points
            if math.isfinite(float(point.value))
        ]
        payload["series"][tag] = series
        values = [point[1] for point in series]
        if values:
            payload["stats"][tag] = {
                "count": len(values),
                "first_step": series[0][0],
                "last_step": series[-1][0],
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
                "first10_mean": statistics.fmean(values[:10]),
                "last10_mean": statistics.fmean(values[-10:]),
            }
print("TB_JSON_BEGIN")
print(json.dumps(payload, sort_keys=True))
print("TB_JSON_END")
PY
else
  echo "tensorboard=MISSING"
fi

echo "SECTION=ARTIFACTS"
if [[ -d "$run_root" ]]; then
  du -sb "$run_root"
  find "$run_root" -maxdepth 2 -type f \
    -printf '%p size=%s mtime=%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort
else
  echo "run_root=MISSING"
fi
if [[ -d "$exp_root/checkpoints" ]]; then
  echo "CHECKPOINTS_BEGIN"
  for checkpoint in "$exp_root"/checkpoints/global_step_*; do
    [[ -d "$checkpoint" ]] || continue
    step=${checkpoint##*_}
    bytes=$(du -sb "$checkpoint" | awk '{print $1}')
    files=$(find "$checkpoint" -type f | wc -l)
    printf 'checkpoint=%s bytes=%s files=%s\n' "$step" "$bytes" "$files"
    complete="$checkpoint/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"
    if [[ -f "$complete" ]]; then
      printf 'completion_%s=' "$step"
      tr -d '\n' < "$complete"
      echo
    else
      echo "completion_${step}=MISSING"
    fi
    for rank in 0 1; do
      state="$checkpoint/actor/sac_components/rlt_trainer_state/checkpoint_rank_${rank}.pt"
      metadata="$checkpoint/actor/sac_components/replay_buffer/rank_${rank}/metadata.json"
      [[ -f "$state" ]] &&
        stat -c "state_${step}_rank${rank} size=%s" "$state" ||
        echo "state_${step}_rank${rank}=MISSING"
      if [[ -f "$metadata" ]]; then
        printf 'replay_%s_rank%s=' "$step" "$rank"
        tr -d '\n' < "$metadata"
        echo
      else
        echo "replay_${step}_rank${rank}=MISSING"
      fi
    done
  done
  echo "CHECKPOINTS_END"
else
  echo "checkpoints=MISSING"
fi

echo "RLT_FORMAL250_STATUS_AUDIT_DONE"
