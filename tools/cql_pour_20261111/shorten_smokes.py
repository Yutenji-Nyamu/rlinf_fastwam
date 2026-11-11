"""Keep full reference/formal horizons; limit lifecycle checks to two chunks."""

import json
import time
from pathlib import Path

from omegaconf import OmegaConf

d = Path("/data/chenyiteng/deployment-20261111/cql-pour/g6")
assert json.loads((d / "status.json").read_text())["phase"] == "reference_eval"
changes = {}
for name in ("smoke", "smoke_resume"):
    assert not (d / f"{name}-identity.json").exists()
    path = d / f"config/{name}.yaml"
    c = OmegaConf.load(path)
    assert c.env.train.max_episode_steps == 100
    assert c.actor.model.num_action_chunks == 10
    assert c.env.train.total_num_envs == 64
    assert c.algorithm.rlt_schedule.max_updates_per_train_step == 16
    for key in ("max_steps_per_rollout_epoch", "max_episode_steps"):
        c.env.train[key] = 20
    OmegaConf.save(c, path)
    changes[name] = {
        "before_steps": 100,
        "after_steps": 20,
        "parallel_envs": 64,
        "expected_chunk_transitions": 128,
        "optimizer_updates": 16,
    }
formal = OmegaConf.load(d / "config/formal.yaml")
assert (
    formal.env.train.max_episode_steps == 400
    and formal.env.eval.max_episode_steps == 400
)
receipt = {
    "time": time.time(),
    "changes": changes,
    "full_400_step_reference_and_formal_unchanged": True,
    "reason": "Two 10-step chunks per environment already exercise real replay insertion, updates, DCP save and resume; full-horizon behavior is covered by the reference evaluation.",
}
(d / "smoke-budget.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
