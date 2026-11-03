# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Bounded GPU7 native reset and one original N10 action chunk before Dojo RLT."""

import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv
from rlinf.models.embodiment.openpi import get_model


def main():
    d = Path(os.environ["DOJO_GRPO_DEPLOYMENT"])
    plan = json.loads((d / "plan.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan["gpu"])
    cfg = OmegaConf.load(d / "config/formal.yaml")
    env = RoboDojoEnv(cfg.env.eval, plan["num_envs"], 0, 1, None)
    start = time.monotonic()
    observation, info = env.reset()
    cfg.actor.model.openpi.use_dsrl = False
    cfg.actor.model.openpi.noise_method = "flow_ode"
    cfg.actor.model.add_q_head = False
    model = get_model(cfg.actor.model).eval().cuda()
    with torch.inference_mode():
        actions, _ = model.predict_action_batch(observation, mode="eval")
    _, _, terms, truncs, _ = env.chunk_step(actions)
    (d / "render-probe.json").write_text(
        json.dumps(
            {
                "elapsed": time.monotonic() - start,
                "environments": plan["num_envs"],
                "actions": 50,
                "terminated": int(terms.any(1).sum()),
                "truncated": int(truncs.any(1).sum()),
                "reset_audit": info["reset_audit"],
            },
            default=str,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
