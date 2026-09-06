set -euo pipefail

PYTHON=/root/autodl-tmp/RLinf/.venv/bin/python
TB_ROOT=/root/autodl-tmp/experiments/ogpo_robotwin_smoke_20260807_v1/tensorboard
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -B - "$TB_ROOT" <<'PY'
import json
import math
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS

root = sys.argv[1]
accumulator = EventAccumulator(root, size_guidance={SCALARS: 0})
accumulator.Reload()
tags = accumulator.Tags().get("scalars", [])
events = {
    tag: [
        {"step": item.step, "value": item.value, "wall_time": item.wall_time}
        for item in accumulator.Scalars(tag)
    ]
    for tag in tags
}

def last(tag):
    assert tag in events and events[tag], tag
    return events[tag][-1]

assert last("train/ogpo/total_online_rows")["step"] == 80
assert last("train/ogpo/total_online_rows")["value"] == 80
assert sum(item["value"] for item in events["train/ogpo/global_inserted_rows"]) == 80
assert sum(item["value"] for item in events["train/ogpo/updates_run"]) == 1
for tag in (
    "train/ogpo/actor_updates",
    "train/ogpo/critic_updates",
    "train/ogpo/policy_version",
):
    assert last(tag)["step"] == 80
    assert last(tag)["value"] == 1
for tag in (
    "train/ogpo/pending_actor_updates",
    "train/ogpo/pending_critic_updates",
):
    assert last(tag)["value"] == 0
for tag in ("train/ogpo/actor_loss", "train/ogpo/critic_loss"):
    assert len(events[tag]) == 1
    assert events[tag][0]["step"] == 80
    assert math.isfinite(events[tag][0]["value"])
assert len(events["eval/num_trajectories"]) == 1
assert events["eval/num_trajectories"][0]["step"] == 80

selected = {
    tag: values
    for tag, values in events.items()
    if tag.startswith("train/ogpo/") or tag.startswith("eval/")
}
print(json.dumps(selected, indent=2, sort_keys=True))
print("OGPO_ROBOTWIN_SMOKE_TENSORBOARD_OK")
PY
