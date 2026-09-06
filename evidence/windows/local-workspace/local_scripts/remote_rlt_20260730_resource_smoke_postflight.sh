#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime
run_root=/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1
experiment=robotwin_adjust_bottle_rlt_stage2_resource_smoke_8env3c_v1
repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python

test "$(cat "$runtime/exit_code.txt")" = 0
test -z "$(
  ps -eo pid=,comm=,args= \
    | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
)"
test -z "$(pgrep -ax raylet || true)"
test -z "$(pgrep -ax gcs_server || true)"
test -z "$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF'
)"

RUNTIME="$runtime" \
RUN_ROOT="$run_root" \
EXPERIMENT="$experiment" \
REPO="$repo" \
  "$python_bin" -B - <<'PY'
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runtime = Path(os.environ["RUNTIME"])
run_root = Path(os.environ["RUN_ROOT"])
experiment = os.environ["EXPERIMENT"]
repo = Path(os.environ["REPO"])

assert hashlib.sha256((runtime / "resolved.yaml").read_bytes()).hexdigest() == (
    "bbcdcdcb22ca93f106dea0786a57086a2e8317caaefd610d18cabe6c4e2ff6aa"
)

seed_path = (
    repo
    / "rlinf/envs/robotwin/seeds/"
    "eval_seeds_adjust_bottle_rlt_periodic20_v1.json"
)
seed_bytes = seed_path.read_bytes()
assert hashlib.sha256(seed_bytes).hexdigest() == (
    "fb9c3353e27b83aad6fe7ff778437d960b084de9d981c2af68615d52769952a7"
)
seeds = json.loads(seed_bytes)["adjust_bottle"]["success_seeds"]
assert len(seeds) == 20
assert len(set(seeds)) == 20

events = list(run_root.rglob("events.out.tfevents.*"))
assert len(events) == 1
acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
acc.Reload()

def values(tag):
    result = [entry.value for entry in acc.Scalars(tag)]
    assert result
    assert all(math.isfinite(value) for value in result)
    return result

assert values("env/num_trajectories") == [8.0, 8.0, 8.0]
assert values("train/rlt/critic_updates_run") == [1600.0, 800.0, 800.0]
assert values("train/rlt/actor_updates_run") == [800.0, 400.0, 400.0]
assert values("train/rlt/update_step") == [0.0, 1600.0, 2400.0]
assert values("eval/num_trajectories") == [20.0]
eval_success = values("eval/success_once")
assert len(eval_success) == 1
assert math.isclose(eval_success[0], 0.1, rel_tol=0.0, abs_tol=1e-7)
for tag in [
    "train/sac/actor_loss",
    "train/sac/critic_loss",
    "train/actor/grad_norm",
    "train/critic/grad_norm",
    "time/step",
]:
    assert len(values(tag)) == 3

checkpoint = (
    run_root
    / experiment
    / "checkpoints/global_step_3/actor/sac_components/"
    "rlt_trainer_state/rlt_trainer_state_complete.json"
)
completion = json.loads(checkpoint.read_text())
assert completion["complete"] is True
assert completion["actor_world_size"] == 2
assert completion["saved_runner_step"] == 3
assert completion["update_step"] == 3200
assert [entry["rank"] for entry in completion["files"]] == [0, 1]
for entry in completion["files"]:
    path = checkpoint.parent / entry["path"]
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert entry["update_step"] == 3200

with (runtime / "resources.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) >= 400

def floats(name):
    return [float(row[name]) for row in rows]

def ints(name):
    return [int(float(row[name])) for row in rows]

gib = 1024**3
summary = {
    "status": "pass",
    "exit_code": 0,
    "started_at": (runtime / "started_at.txt").read_text().strip(),
    "finished_at": (runtime / "finished_at.txt").read_text().strip(),
    "resource_rows": len(rows),
    "resolved_sha256": hashlib.sha256(
        (runtime / "resolved.yaml").read_bytes()
    ).hexdigest(),
    "seed_bank_sha256": hashlib.sha256(seed_bytes).hexdigest(),
    "seed_count": len(seeds),
    "seed_unique_count": len(set(seeds)),
    "train_episodes": 24,
    "macro_transitions": 472,
    "critic_updates": 3200,
    "actor_updates": 1600,
    "eval_episodes": 20,
    "eval_success_numerator": round(eval_success[0] * 20),
    "eval_success_rate": eval_success[0],
    "checkpoint_complete": True,
    "checkpoint_update_step": completion["update_step"],
    "gpu0_peak_mib": max(ints("gpu0_used_mib")),
    "gpu1_peak_mib": max(ints("gpu1_used_mib")),
    "gpu0_peak_util_pct": max(ints("gpu0_util_pct")),
    "gpu1_peak_util_pct": max(ints("gpu1_util_pct")),
    "cgroup_current_peak_gib": max(floats("cgroup_current_bytes")) / gib,
    "cgroup_anon_peak_gib": max(floats("cgroup_anon_bytes")) / gib,
    "cgroup_file_peak_gib": max(floats("cgroup_file_bytes")) / gib,
    "matched_rss_peak_gib": max(floats("matched_total_rss_kib")) / 1024**2,
    "env_rss_peak_gib": max(floats("env_rss_kib")) / 1024**2,
    "host_available_min_gib": min(floats("host_available_bytes")) / gib,
    "cgroup_high_delta": ints("cgroup_high_events")[-1]
    - ints("cgroup_high_events")[0],
    "cgroup_max_delta": ints("cgroup_max_events")[-1]
    - ints("cgroup_max_events")[0],
    "cgroup_oom_delta": ints("cgroup_oom_events")[-1]
    - ints("cgroup_oom_events")[0],
    "cgroup_oom_kill_delta": ints("cgroup_oom_kill_events")[-1]
    - ints("cgroup_oom_kill_events")[0],
    "actor_loss": values("train/sac/actor_loss"),
    "critic_loss": values("train/sac/critic_loss"),
    "actor_grad_norm": values("train/actor/grad_norm"),
    "critic_grad_norm": values("train/critic/grad_norm"),
    "step_seconds": values("time/step"),
}
started = datetime.fromisoformat(summary["started_at"])
finished = datetime.fromisoformat(summary["finished_at"])
summary["wall_seconds"] = (finished - started).total_seconds()
summary["run_bytes"] = sum(
    path.stat().st_size for path in run_root.rglob("*") if path.is_file()
)

assert summary["gpu0_peak_mib"] < 40960
assert summary["gpu1_peak_mib"] < 40960
assert summary["cgroup_anon_peak_gib"] < 100
assert summary["cgroup_high_delta"] == 0
assert summary["cgroup_oom_delta"] == 0
assert summary["cgroup_oom_kill_delta"] == 0
assert max(summary["actor_grad_norm"]) < 10
assert max(summary["critic_grad_norm"]) < 10

output = runtime / "postflight_summary.json"
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if grep -Eiq \
  'CUDA out of memory|NCCL[^[:cntrl:]]*(fatal|error)|RayActorError|nan([^[:alpha:]]|$)' \
  "$runtime/driver.log"
then
  printf '%s\n' 'fatal_pattern_found' >&2
  exit 1
fi
printf '%s\n' RLT_RESOURCE_SMOKE_8ENV_POSTFLIGHT_OK
