#!/usr/bin/env bash
set -u

root=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2
run="$root/run"

python3 - "$run/driver.log" "$run/resource.csv" <<'PY'
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
resource_path = Path(sys.argv[2])
raw = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
raw = raw.replace("\x08", "")
anchors = list(re.finditer(r"Global Step:\s*(\d+)/200", raw))
rows = []
num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def metrics(text):
    out = {}
    for key, val in re.findall(r"([A-Za-z][A-Za-z0-9_./-]*)=(%s)" % num, text):
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out

for idx, match in enumerate(anchors):
    end = anchors[idx + 1].start() if idx + 1 < len(anchors) else len(raw)
    block = raw[match.start():end]
    marker = block.find("Evaluation")
    if marker >= 0:
        train_text, eval_text = block[:marker], block[marker:]
    else:
        train_text, eval_text = block, ""
    train = metrics(train_text)
    post = metrics(eval_text)
    combined = dict(train)
    for key, value in post.items():
        if key not in {"reward", "success_once", "success_at_end"}:
            combined[key] = value
    rows.append({
        "step": int(match.group(1)),
        "train": train,
        "eval": {k: v for k, v in post.items() if k in {"reward", "success_once", "success_at_end"}},
        "all": combined,
    })

def values(key, subset=None):
    source = rows if subset is None else subset
    return [r["all"][key] for r in source if key in r["all"] and math.isfinite(r["all"][key])]

def mean(xs):
    return statistics.fmean(xs) if xs else None

def tail_mean(xs, n):
    return statistics.fmean(xs[-n:]) if xs else None

success = [r["train"].get("success_once") for r in rows if "success_once" in r["train"]]
first_update = next((r["step"] for r in rows if "sac/actor_loss" in r["all"]), None)
planned = [r["all"].get("sac/planned_optimizer_updates", 0.0) for r in rows]
evals = [
    {
        "step": r["step"],
        "success_once": r["eval"].get("success_once"),
        "success_at_end": r["eval"].get("success_at_end"),
        "reward": r["eval"].get("reward"),
    }
    for r in rows if r["eval"]
]

metric_keys = [
    "sac/actor_loss", "sac/critic_loss", "sac/alpha", "sac/alpha_loss",
    "actor/entropy", "actor/grad_norm", "critic/grad_norm", "alpha/grad_norm",
    "actor/q_pi", "critic/q_data", "actor/q_value_0", "actor/q_value_1",
    "sac/global_new_transitions", "sac/global_resident_transitions",
    "sac/planned_optimizer_updates", "actor/run_training", "step",
]
metric_summary = {}
last = rows[-1] if rows else None
for key in metric_keys:
    xs = values(key)
    metric_summary[key] = {
        "count": len(xs),
        "first": xs[0] if xs else None,
        "last": last["all"].get(key) if last else None,
        "mean": mean(xs),
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
        "last10_mean": tail_mean(xs, 10),
    }

with resource_path.open(newline="", encoding="utf-8") as f:
    resource = list(csv.DictReader(f))

def nums(field):
    out = []
    for row in resource:
        try:
            out.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return out

resource_summary = {
    "rows": len(resource),
    "start": resource[0]["timestamp"] if resource else None,
    "end": resource[-1]["timestamp"] if resource else None,
    "driver_alive_values": sorted(set(nums("driver_alive"))),
}
for field in ["gpu6_used_mib", "gpu7_used_mib", "gpu6_util_pct", "gpu7_util_pct"]:
    xs = nums(field)
    resource_summary[field] = {
        "min": min(xs) if xs else None,
        "median": statistics.median(xs) if xs else None,
        "mean": mean(xs),
        "p95": sorted(xs)[min(len(xs)-1, int(0.95*(len(xs)-1)))] if xs else None,
        "max": max(xs) if xs else None,
        "last": xs[-1] if xs else None,
    }
avail = nums("host_mem_available_kib")
resource_summary["host_available_gib"] = {
    "min": min(avail) / 1024 / 1024 if avail else None,
    "max": max(avail) / 1024 / 1024 if avail else None,
    "last": avail[-1] / 1024 / 1024 if avail else None,
}
for field in ["cgroup_oom", "cgroup_oom_kill", "mem_psi_some_avg10", "mem_psi_full_avg10"]:
    xs = nums(field)
    resource_summary[field] = {"max": max(xs) if xs else None, "last": xs[-1] if xs else None}

out = {
    "step_count": len(rows),
    "first_step": rows[0]["step"] if rows else None,
    "last_complete_step": rows[-1]["step"] if rows else None,
    "first_optimizer_step": first_update,
    "train_success": {
        "count": len(success),
        "mean_all": mean(success),
        "mean_warmup_1_12": mean(success[:12]),
        "mean_learned_13_latest": mean(success[12:]),
        "last": success[-1] if success else None,
        "last5_mean": tail_mean(success, 5),
        "last10_mean": tail_mean(success, 10),
        "last20_mean": tail_mean(success, 20),
        "episode_successes_all": sum(success) * 4 if success else None,
        "episodes_all": len(success) * 4,
    },
    "evals": evals,
    "cumulative_planned_optimizer_updates": sum(planned),
    "metric_summary": metric_summary,
    "resource_summary": resource_summary,
}
print(json.dumps(out, indent=2, sort_keys=True))
PY

printf '%s\n' '--- exact ERROR lines ---'
tr '\r' '\n' < "$run/driver.log" | grep -in -C 1 'ERROR' | head -n 100 || true

printf '%s\n' '--- checkpoint bytes and small manifests ---'
for d in "$run/dsrl-current-formal-200c-v2/checkpoints"/global_step_*; do
  [ -d "$d" ] || continue
  du -sb "$d"
  find "$d" -type f \( -name '*.json' -o -name 'complete*' -o -name 'manifest*' \) -size -256k -print -exec sed -n '1,160p' {} \; 2>/dev/null || true
done

printf '%s\n' '--- top-level directories ---'
find "$root" -maxdepth 3 -type d -printf '%p\n' | sort | head -n 240
