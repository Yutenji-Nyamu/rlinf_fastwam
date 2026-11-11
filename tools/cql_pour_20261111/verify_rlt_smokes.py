"""Inspect completed real checkpoint/resume artifacts, without GPU allocation."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

b = Path("/data/chenyiteng/deployment-20261111/cql-pour-56/g6")
r = Path("/home/chenyiteng/results/cql-pour-20261111/g6-56")
result = {}
phase_weights = {}
for phase, step in [("smoke", 1), ("smoke_resume", 2)]:
    phase_deploy = (
        Path("/data/chenyiteng/deployment-20261111/cql-pour/g6")
        if phase == "smoke"
        else b
    )
    phase_run = (
        Path("/home/chenyiteng/results/cql-pour-20261111/g6") if phase == "smoke" else r
    )
    assert (phase_deploy / f"{phase}-complete.json").exists(), f"{phase} not complete"
    cp = phase_run / phase / phase / "checkpoints" / f"global_step_{step}"
    assert (cp / "dojo-complete.json").exists()
    model = cp / "actor/model_state_dict/full_weights.pt"
    manifest = json.loads(
        (cp / "actor/sac_components/rlt_trainer_state/complete.json").read_text()
    )
    state = torch.load(
        cp / "actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt",
        map_location="cpu",
        weights_only=True,
    )
    weights = torch.load(model, map_location="cpu", weights_only=True)
    assert all(torch.isfinite(t).all() for t in weights.values())
    phase_weights[phase] = weights
    scalars = {}
    for path in (phase_run / phase).rglob("events.out.tfevents.*"):
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for key in acc.Tags()["scalars"]:
            if key.startswith("train/") and any(
                x in key
                for x in ("loss", "grad_norm", "update_step", "ready_for_online")
            ):
                values = [x.value for x in acc.Scalars(key)]
                assert all(math.isfinite(x) for x in values), key
                scalars[key] = values
    assert scalars
    result[phase] = {
        "checkpoint": str(cp),
        "manifest": manifest,
        "update_step": state["update_step"],
        "total_transitions": state["local_total_transitions_added"],
        "warmup_ready_at": state["global_warmup_ready_total_transitions"],
        "contract": json.loads(state["rlt_resume_contract"]),
        "policy_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "all_policy_tensors_finite": True,
        "metrics": scalars,
        "files": [
            {"path": str(x.relative_to(cp)), "bytes": x.stat().st_size}
            for x in cp.rglob("*")
            if x.is_file()
        ],
    }
a, c = result["smoke"], result["smoke_resume"]
assert a["update_step"] == 16 and c["update_step"] == 32
assert c["total_transitions"] > a["total_transitions"] > 0
assert c["warmup_ready_at"] == a["warmup_ready_at"]
assert c["contract"] == a["contract"]
assert c["policy_sha256"] != a["policy_sha256"]
before, after = phase_weights["smoke"], phase_weights["smoke_resume"]
assert before.keys() == after.keys()
deltas = {
    key: float((after[key].float() - before[key].float()).abs().max())
    for key in before
    if before[key].numel()
}
assert any(value > 0 for value in deltas.values())
result["policy_parameter_delta"] = {
    "tensors_changed": sum(value > 0 for value in deltas.values()),
    "tensors_compared": len(deltas),
    "max_abs": max(deltas.values()),
}
result["passed"] = True
result["scope"] = (
    "real native64 collection and updates, then native56 DCP resume and further updates; disposable smokes are not formal training data"
)
(b / "smoke-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
