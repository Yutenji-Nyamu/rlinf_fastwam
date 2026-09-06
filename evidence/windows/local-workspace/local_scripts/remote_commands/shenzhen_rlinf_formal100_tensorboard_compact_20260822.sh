#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -B - "$RUN" <<'PY'
import json
import math
import pathlib
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = pathlib.Path(sys.argv[1])
events = sorted(run.rglob("events.out.tfevents.*"))
print("EVENTS=" + json.dumps([{"path": str(p), "bytes": p.stat().st_size} for p in events]))
for path in events:
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = sorted(acc.Tags().get("scalars", []))
    data = {
        tag: [{"s": int(x.step), "v": float(x.value), "t": float(x.wall_time)} for x in acc.Scalars(tag)]
        for tag in tags
    }
    summary = []
    for tag, vals in data.items():
        numbers = [x["v"] for x in vals]
        summary.append({
            "tag": tag, "n": len(vals),
            "s0": vals[0]["s"] if vals else None, "s1": vals[-1]["s"] if vals else None,
            "v0": numbers[0] if numbers else None, "v1": numbers[-1] if numbers else None,
            "min": min(numbers) if numbers else None, "max": max(numbers) if numbers else None,
            "finite": all(math.isfinite(x) for x in numbers),
        })
    print("SUMMARY=" + json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))

    exact = {
        "env/success_once", "env/return", "env/reward", "env/num_trajectories",
        "eval/success_once", "eval/success_at_end", "eval/return", "eval/num_trajectories",
        "rollout/advantages_mean", "rollout/advantages_min", "rollout/advantages_max",
        "rollout/returns_mean", "rollout/returns_min", "rollout/returns_max",
        "train/actor/approx_kl", "train/actor/clip_fraction", "train/actor/grad_norm",
        "train/actor/policy_loss", "train/actor/total_loss", "train/actor/ratio",
        "train/critic/value_loss", "train/critic/explained_variance", "train/critic/value_clip_ratio",
        "time/generate_rollouts", "time/actor_training", "time/sync_weights", "time/step",
    }
    selected = {tag: vals for tag, vals in data.items() if tag in exact}
    print("SERIES=" + json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False))

    anchor = data.get("env/success_once", [])
    deltas = [anchor[i]["t"] - anchor[i - 1]["t"] for i in range(1, len(anchor))]
    print("WALLTIME=" + json.dumps({
        "n": len(anchor),
        "start": anchor[0]["t"] if anchor else None,
        "end": anchor[-1]["t"] if anchor else None,
        "delta_s": deltas,
        "mean_s": statistics.mean(deltas) if deltas else None,
        "median_s": statistics.median(deltas) if deltas else None,
        "p90_s": sorted(deltas)[int(0.9 * (len(deltas) - 1))] if deltas else None,
    }, separators=(",", ":"), allow_nan=False))
PY

printf 'LOG_STATUS='; grep -E 'Global Step:' "$RUN/driver.log" | tail -n 1 | sed -E 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g'
printf 'LATEST_PROGRESS='; grep -E 'Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$RUN/driver.log" | tail -n 1 | sed -E 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g'
printf 'FATAL_COUNT='; grep -Ec 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left' "$RUN/driver.log" || true
