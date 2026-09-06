#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
PID="$(cat "$RUN/driver.pid")"

printf '=== LIVE ===\n'
printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'driver_pid=%s\n' "$PID"
if kill -0 "$PID" 2>/dev/null; then printf 'driver_alive=yes\n'; else printf 'driver_alive=no\n'; fi
ps -p "$PID" -o lstart=,etime=,stat=,rss=,vsz=,args=
printf 'driver_cgroup='; awk -F: '$1 == "0" {print $3}' "/proc/$PID/cgroup"

printf '=== HOST_MEMORY ===\n'
grep -E '^(MemTotal|MemFree|MemAvailable|Cached|SReclaimable|Shmem|SwapTotal|SwapFree):' /proc/meminfo
free -h
CGROUP_REL="$(awk -F: '$1 == "0" {print $3}' "/proc/$PID/cgroup")"
CGROUP_BASE="/sys/fs/cgroup${CGROUP_REL}"
for name in memory.current memory.peak memory.high memory.max memory.events; do
  if [[ -r "$CGROUP_BASE/$name" ]]; then
    printf -- '--- cgroup/%s ---\n' "$name"
    cat "$CGROUP_BASE/$name"
  fi
done
printf 'user_process_rss_kib='; ps -u chenyiteng -o rss= | awk '{s+=$1} END {print s+0}'
printf 'envworker_rss_kib='; ps -u chenyiteng -o rss=,args= | awk '/ray::EnvWorker/ {s+=$1} END {print s+0}'

printf '=== GPUS ===\n'
nvidia-smi -i 4,5,6,7 \
  --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format=csv,noheader,nounits
printf -- '--- compute apps ---\n'
nvidia-smi -i 4,5,6,7 \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits || true

printf '=== RUN_AND_ARTIFACTS ===\n'
du -sh "$RUN"
df -h "$RUN"
find "$RUN" -type f -printf '%f\t%s\n' | awk -F '\t' '
  {n++; b+=$2; ext=$1; sub(/^.*\./,".",ext); if ($1 !~ /\./) ext="[no_ext]"; c[ext]++; s[ext]+=$2}
  END {printf "files_total=%d bytes_total=%d\n",n,b; for (e in c) printf "filetype=%s count=%d bytes=%d\n",e,c[e],s[e]}' | sort
printf -- '--- top-level inventory ---\n'
find "$RUN" -mindepth 1 -maxdepth 3 -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' | sort -k4,4

printf '=== CHECKPOINTS ===\n'
while IFS= read -r -d '' ckpt; do
  printf 'checkpoint=%s bytes=' "$ckpt"
  du -sb "$ckpt" | awk '{print $1}'
  find "$ckpt" -type f -printf '%s\t%p\n' | sort -nr | head -n 20
  printf 'metadata_count='; find "$ckpt" -type f -name '.metadata' | wc -l
  printf 'distcp_count='; find "$ckpt" -type f -name '*.distcp' | wc -l
  printf 'full_weights_count='; find "$ckpt" -type f -name 'full_weights.pt' | wc -l
done < <(find "$RUN" -type d -path '*/checkpoints/global_step_*' -print0 | sort -zV)

printf '=== VIDEOS ===\n'
for split in train eval; do
  base="$RUN/video/$split"
  if [[ -d "$base" ]]; then
    printf 'split=%s count=' "$split"; find "$base" -type f -name '*.mp4' | wc -l
    printf 'split=%s bytes=' "$split"; find "$base" -type f -name '*.mp4' -printf '%s\n' | awk '{s+=$1} END {print s+0}'
    for seed in "$base"/seed_*; do
      [[ -d "$seed" ]] || continue
      printf 'seed_dir=%s count=' "$seed"; find "$seed" -maxdepth 1 -type f -name '*.mp4' | wc -l
      find "$seed" -maxdepth 1 -type f -name '*.mp4' -printf '%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort -V | tail -n 3
    done
  fi
done

printf '=== TENSORBOARD ===\n'
PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -B - "$RUN" <<'PY'
import json
import math
import pathlib
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = pathlib.Path(sys.argv[1])
events = sorted(run.rglob("events.out.tfevents.*"))
print(json.dumps({"event_files": [{"path": str(p), "bytes": p.stat().st_size} for p in events]}))
for path in events:
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = sorted(acc.Tags().get("scalars", []))
    tag_summary = []
    series = {}
    for tag in tags:
        vals = acc.Scalars(tag)
        numbers = [float(x.value) for x in vals]
        tag_summary.append({
            "tag": tag,
            "count": len(vals),
            "first_step": int(vals[0].step) if vals else None,
            "last_step": int(vals[-1].step) if vals else None,
            "first": numbers[0] if numbers else None,
            "last": numbers[-1] if numbers else None,
            "min": min(numbers) if numbers else None,
            "max": max(numbers) if numbers else None,
            "finite": all(math.isfinite(v) for v in numbers),
        })
        series[tag] = [{"step": int(x.step), "value": float(x.value), "wall_time": float(x.wall_time)} for x in vals]
    print("TAG_SUMMARY_JSON=" + json.dumps(tag_summary, sort_keys=True, allow_nan=False))

    wanted_exact = {
        "env/success_once", "eval/success_once", "env/num_trajectories", "eval/num_trajectories",
        "train/actor/approx_kl", "train/actor/clip_fraction", "train/actor/grad_norm",
        "train/actor/policy_loss", "train/actor/total_loss", "train/actor/ratio",
        "train/critic/value_loss", "train/critic/explained_variance", "train/critic/value_clip_ratio",
        "train/advantages/mean", "train/advantages/max", "train/advantages/min",
        "train/returns/mean", "train/returns/max", "train/returns/min",
        "time/generate_rollouts", "time/actor_training", "time/sync_weights", "time/global_step",
    }
    wanted_keywords = ("success", "trajectory", "approx_kl", "clip_fraction", "grad_norm", "value_loss", "explained_variance", "global_step")
    selected = {
        tag: values for tag, values in series.items()
        if tag in wanted_exact or any(k in tag.lower() for k in wanted_keywords)
    }
    print("SELECTED_SERIES_JSON=" + json.dumps(selected, sort_keys=True, allow_nan=False))

    # One row per completed logger step for the highest-value training signals.
    columns = [tag for tag in (
        "env/success_once", "train/actor/approx_kl", "train/actor/clip_fraction",
        "train/actor/grad_norm", "train/actor/policy_loss", "train/critic/value_loss",
        "train/critic/explained_variance", "time/global_step",
    ) if tag in series]
    steps = sorted({item["step"] for tag in columns for item in series[tag]})
    lookup = {tag: {item["step"]: item["value"] for item in series[tag]} for tag in columns}
    print("STEP_TABLE_COLUMNS=" + json.dumps(["event_step"] + columns))
    for step in steps:
        print("STEP_ROW_JSON=" + json.dumps([step] + [lookup[tag].get(step) for tag in columns], allow_nan=False))

    # Wall-time differences between completed logger steps.
    anchor = None
    for candidate in ("env/success_once", "train/actor/total_loss", "train/critic/value_loss"):
        if candidate in series:
            anchor = candidate
            break
    if anchor:
        pts = series[anchor]
        deltas = [pts[i]["wall_time"] - pts[i-1]["wall_time"] for i in range(1, len(pts))]
        print("STEP_WALLTIME_JSON=" + json.dumps({
            "anchor": anchor,
            "count": len(pts),
            "first_wall_time": pts[0]["wall_time"],
            "last_wall_time": pts[-1]["wall_time"],
            "deltas_s": deltas,
            "mean_delta_s": statistics.mean(deltas) if deltas else None,
            "median_delta_s": statistics.median(deltas) if deltas else None,
        }, allow_nan=False))
PY

printf '=== LOG_HEALTH_AND_PROGRESS ===\n'
printf 'driver_log_bytes='; stat -c %s "$RUN/driver.log"
printf 'fatal_count='; grep -Ec 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left' "$RUN/driver.log" || true
grep -nE 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left' "$RUN/driver.log" | tail -n 30 || true
printf 'nan_inf_suspicious_count='; grep -Eic '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$RUN/driver.log" || true
printf -- '--- latest completed/progress markers ---\n'
grep -E 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$RUN/driver.log" | tail -n 45 || true
printf -- '--- latest scalar table tail ---\n'
tail -n 180 "$RUN/driver.log"
