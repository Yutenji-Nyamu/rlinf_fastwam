# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Read completed smoke tensors and prove direction, finite update and release."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from common import actors, gpus, read, same, save
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument("--plan", required=True)
p.add_argument("--lane", choices=("signed", "positive"), required=True)
a = p.parse_args()
plan = read(a.plan)
q = plan["runs"][a.lane]
run, rt = Path(q["run"]), Path(q["runtime"])
finished = read(rt / "finished.json")
assert finished["exit_code"] == 0 and finished["cleanup_error"] is None
assert not same(read(rt / "started.json")["identity"])
assert not actors(plan, q["namespace"])
snapshot = gpus()
assert all(not snapshot[g]["processes"] for g in q["gpus"])
paths = sorted((run / "dvac_train").glob("actor_rank*/runner_step_0000.pt"))
assert len(paths) == 2, paths
data = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
assert {row["actor_rank"] for row in data} == {0, 1}
full = {
    key: torch.cat([row[key] for row in data], dim=1)
    for key in (
        "variance",
        "weights",
        "advantages",
        "loss_mask",
        "local_factors",
        "chunk_factors",
        "chunk_scores",
        "chunk_contributions",
        "eligible_mask",
    )
}
ids = torch.cat([row["group_ids"] for row in data])
u, weights = full["variance"], full["weights"]
adv = full["advantages"].squeeze(-1)
valid = full["loss_mask"].squeeze(-1).bool()
eligible = full["eligible_mask"].squeeze(-1).bool()
assert weights.shape == (4, 256, 50), weights.shape
assert torch.isfinite(weights).all() and (weights > 0).all()
assert torch.equal(weights[~eligible], torch.ones_like(weights[~eligible]))
direction_counts = {}
for sign in (1, -1):
    selected = eligible & (adv * sign > 0)
    direction_counts[str(sign)] = int(selected.sum())
    if not selected.any():
        continue
    order = u[selected].argsort(-1)
    ordered = weights[selected].gather(-1, order)
    delta = ordered.diff(dim=-1) * sign
    assert (delta >= -2e-6).all(), "Action weights have the wrong U direction"
    assert (delta > 2e-6).any(), "No nonconstant action weighting"
    for group in ids.unique():
        mask = selected & (ids == group).unsqueeze(0)
        if not mask.any():
            continue
        scores = full["chunk_scores"].squeeze(-1)[mask]
        factors = full["chunk_factors"].squeeze(-1)[mask]
        assert (factors[scores.argsort()].diff() * sign >= -2e-6).all()
        mass = full["chunk_contributions"].squeeze(-1)[mask]
        mean = (weights.mean(-1)[mask] * mass).sum() / mass.sum()
        torch.testing.assert_close(mean, torch.ones_like(mean), atol=2e-6, rtol=0.0)
assert direction_counts["1"] > 0
if a.lane == "signed":
    assert direction_counts["-1"] > 0
else:
    assert torch.equal(weights[adv < 0], torch.ones_like(weights[adv < 0]))
scalars = {}
for path in run.rglob("events.out.tfevents.*"):
    events = EventAccumulator(str(path)).Reload()
    for key in events.Tags()["scalars"]:
        values = events.Scalars(key)
        if values:
            scalars[key] = values[-1].value
assert scalars["env/num_trajectories"] == 256
for key in ("train/actor/grad_norm", "train/actor/total_loss"):
    assert math.isfinite(scalars[key])
assert scalars["train/actor/grad_norm"] > 0
start = read(rt / "started.json")["time"]
scope = Path(a.plan).parent / "scope"
receipts = [read(path) for path in (scope / "receipts").glob("*-renderer_bound.json")]
receipts = [r for r in receipts if r["time"] >= start]
assert {r["physical_gpu"] for r in receipts} == set(q["gpus"])
result = {
    "time": time.time(),
    "lane": a.lane,
    "passed": True,
    "finished": finished,
    "elapsed_seconds": finished["time"] - start,
    "trajectory_budget": 256,
    "update_epoch": 2,
    "rank_shapes": [list(row["weights"].shape) for row in data],
    "weight_min": float(weights[valid].min()),
    "weight_max": float(weights[valid].max()),
    "direction_chunks": direction_counts,
    "within_chunk_direction_passed": True,
    "between_chunk_direction_passed": True,
    "sign_pool_mean_one_passed": True,
    "excluded_identity_passed": True,
    "metrics": data[0]["metrics"],
    "scalars": {
        k: v
        for k, v in scalars.items()
        if k.startswith(("env/", "train/actor/", "time/"))
    },
    "namespace_clear": True,
    "compute_and_graphics_clear": True,
    "renderer_gpu_uuids": sorted({r["gpu_uuid"] for r in receipts}),
    "formal_training": False,
}
save(rt / "smoke-validation.json", result)
print(json.dumps(result))
