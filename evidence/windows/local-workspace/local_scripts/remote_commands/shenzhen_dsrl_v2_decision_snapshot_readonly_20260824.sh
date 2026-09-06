#!/usr/bin/env bash
set -u

root=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run
event=$(find "$root/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents*' | head -n 1)
py=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

"$py" - "$event" "$root/resource.csv" <<'PY'
import csv
import json
import math
import statistics
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event, resource_path = sys.argv[1:]
ea = EventAccumulator(event, size_guidance={"scalars": 0})
ea.Reload()

def vals(tag):
    return [(x.step + 1, float(x.value), float(x.wall_time)) for x in ea.Scalars(tag)]

def mean_tail(items, n):
    values = [x[1] for x in items[-n:]]
    return statistics.fmean(values) if values else None

success = vals("env/success_once")
evals = vals("eval/success_once")
new = vals("train/sac/global_new_transitions")
planned = vals("train/sac/planned_optimizer_updates")
resident = vals("train/sac/global_resident_transitions")
step_time = vals("time/step")
train_time = vals("time/actor/run_training")
rollout_time = vals("time/generate_rollouts")

metrics = {}
for tag in [
    "train/sac/actor_loss", "train/sac/critic_loss", "train/sac/alpha",
    "train/actor/entropy", "train/actor/grad_norm", "train/critic/grad_norm",
    "train/actor/q_pi", "train/critic/q_data",
]:
    series = vals(tag)
    metrics[tag] = {
        "last": series[-1][1],
        "last10_mean": mean_tail(series, 10),
        "last20_mean": mean_tail(series, 20),
        "finite": all(math.isfinite(x[1]) for x in series),
    }

with open(resource_path, newline="", encoding="utf-8") as f:
    rr = list(csv.DictReader(f))

out = {
    "last_global_step": success[-1][0],
    "train_success": {
        "all_mean": statistics.fmean(x[1] for x in success),
        "learned_step13_latest_mean": statistics.fmean(x[1] for x in success[12:]),
        "last5": mean_tail(success, 5),
        "last10": mean_tail(success, 10),
        "last20": mean_tail(success, 20),
        "last30": mean_tail(success, 30),
        "last10_points": [[x[0], x[1]] for x in success[-10:]],
    },
    "evals": [[x[0], x[1]] for x in evals],
    "eval_last3_mean": mean_tail(evals, 3),
    "eval_last5_mean": mean_tail(evals, 5),
    "global_resident": resident[-1][1],
    "cumulative_planned_updates": sum(x[1] for x in planned),
    "last10_new_transitions_mean": mean_tail(new, 10),
    "timing_seconds": {
        "last10_step_mean": mean_tail(step_time, 10),
        "last10_train_mean": mean_tail(train_time, 10),
        "last10_rollout_mean": mean_tail(rollout_time, 10),
    },
    "metrics": metrics,
    "resource": {
        "rows": len(rr),
        "host_available_gib_last": float(rr[-1]["host_mem_available_kib"]) / 1024 / 1024,
        "host_available_gib_min": min(float(x["host_mem_available_kib"]) for x in rr) / 1024 / 1024,
        "gpu6_mib_last": float(rr[-1]["gpu6_used_mib"]),
        "gpu7_mib_last": float(rr[-1]["gpu7_used_mib"]),
        "oom_max": max(float(x["cgroup_oom"]) for x in rr),
        "oom_kill_max": max(float(x["cgroup_oom_kill"]) for x in rr),
        "psi_some_max": max(float(x["mem_psi_some_avg10"]) for x in rr),
        "psi_full_max": max(float(x["mem_psi_full_avg10"]) for x in rr),
    },
}
print(json.dumps(out, indent=2, sort_keys=True))
PY

printf '%s\n' '--- lightweight source files ---'
for f in \
  "$root/driver.log" "$root/resource.csv" "$root/resolved.yaml" "$root/command.txt" \
  "$root/launch_manifest.txt" "$root/started_at.txt" "$root/exit_code.txt" "$event"; do
  [ -e "$f" ] && stat -c '%s|%n' "$f" || true
done
printf '%s\n' '--- checkpoint sizes ---'
for d in "$root/dsrl-current-formal-200c-v2/checkpoints"/global_step_*; do
  [ -d "$d" ] && du -sb "$d"
done
