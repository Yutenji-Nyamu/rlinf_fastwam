# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Check the older accepted DSRL model code against the accepted Dojo output."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.models.model import Observation

from rlinf.models.embodiment.openpi import get_model


def main():
    d = Path(os.environ["DOJO_GRPO_DEPLOYMENT"])
    p = json.loads((d / "plan.json").read_text())
    reference = Path(p["parity_source"])
    while not (reference / "torch-probe.npz").is_file():
        reference = Path(
            json.loads((reference / "parity.json").read_text())[
                "reused_verified_source"
            ]
        )
    raw = np.load(p["probe_observation"])
    model = (
        get_model(OmegaConf.load(d / "config/formal.yaml").actor.model).eval().cuda()
    )
    allowed = (
        "dsrl_action_noise_net.",
        "actor_image_encoder.",
        "actor_state_encoder.",
        "critic_image_encoder.",
        "critic_state_encoder.",
        "q_head.",
    )
    trainable = [
        name for name, value in model.named_parameters() if value.requires_grad
    ]
    assert trainable and all(name.startswith(allowed) for name in trainable), trainable
    noise = np.random.RandomState(1103).normal(size=(1, 50, 32)).astype(np.float32)

    def forward(n):
        obs = {
            k: torch.from_numpy(np.repeat(raw[k][:1], n, 0))
            for k in ("states", "main_images", "wrist_images")
        }
        obs.update(
            task_descriptions=[str(raw["task_descriptions"][0])] * n,
            extra_view_images=None,
        )
        processed = model.input_transform(model.obs_processor(obs), transpose=False)
        sample = Observation.from_dict(model.precision_processor(processed))
        with torch.inference_mode():
            out = model.sample_actions(
                sample,
                noise=torch.from_numpy(np.repeat(noise, n, 0)).cuda(),
                mode="eval",
                compute_values=False,
            )
            action = model.output_transform(
                {"actions": out["actions"], "state": sample.state}
            )["actions"]
        return action.detach().float().cpu().numpy()

    one = forward(1)
    error = np.abs(one[0] - np.load(reference / "torch-probe.npz")["actions"])
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    batch = forward(p["num_envs"])
    elapsed = time.monotonic() - start
    delta = batch - one
    result = {
        "accepted_reference": str(reference),
        "same_observation_noise_rmse": float(np.sqrt((error**2).mean())),
        "max_abs": float(error.max()),
        "batch": p["num_envs"],
        "batch_rmse": float(np.sqrt((delta**2).mean())),
        "seconds": elapsed,
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "frozen_vla": True,
        "trainable_names": trainable,
        "strict_load": model.dsrl_checkpoint_load_report,
    }
    (d / "model-contract.json").write_text(json.dumps(result, indent=2))
    assert (
        result["same_observation_noise_rmse"] < 0.02
        and result["max_abs"] < 0.15
        and result["batch_rmse"] < 0.02
    ), result
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("trainable_names", "strict_load")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
