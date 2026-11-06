# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""One 8-episode frozen-reference evaluation plus action adapter parity."""

import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main(directory):
    d = Path(directory)
    p = json.loads((d / "plan.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p["gpu"])
    from rlinf.envs import get_env_cls
    from rlinf.models.embodiment.openpi import get_model

    c = OmegaConf.load(d / "config/formal.yaml")
    model = get_model(c.rollout.rlt_feature_model).eval().cuda()
    ec = copy.deepcopy(c.env.eval)
    ec.total_num_envs = 8
    ec.auto_reset = False
    ec.ignore_terminations = False
    env = get_env_cls(ec.env_type, ec)(ec, 8, 0, 1, None)
    obs, _ = env.reset()
    obs.setdefault("extra_view_images", None)
    with torch.no_grad():
        torch.manual_seed(8811)
        torch.cuda.manual_seed_all(8811)
        features, context = model.extract_rlt_obs(obs, return_decode_context=True)
        decoded = model.decode_rlt_action(features["ref_chunk"], context).cpu().float()
        torch.manual_seed(8811)
        torch.cuda.manual_seed_all(8811)
        native, _ = model.predict_action_batch(obs, mode="eval")
        native = torch.as_tensor(native).cpu().float()
        delta = (decoded - native).abs()
        assert torch.isfinite(decoded).all() and float(delta.max()) < 1e-5
    success = torch.zeros(8, dtype=torch.bool)
    done = success.clone()
    start = time.time()
    chunk = int(c.actor.model.num_action_chunks)
    for step in range(int(ec.max_episode_steps) // chunk):
        obs.setdefault("extra_view_images", None)
        with torch.no_grad():
            actions, _ = model.predict_action_batch(obs, mode="eval")
        next_obs, _, term, trunc, _ = env.chunk_step(actions)
        term = torch.as_tensor(term).bool().reshape(8, -1).any(-1).cpu()
        trunc = torch.as_tensor(trunc).bool().reshape(8, -1).any(-1).cpu()
        success |= term & ~done
        done |= term | trunc
        obs = next_obs[-1]
        if done.all():
            break
    result = {
        "time": time.time(),
        "successes": int(success.sum()),
        "episodes": 8,
        "layout_ids": env.reset_state_ids.tolist(),
        "seconds": time.time() - start,
        "canonical_decode_max_abs": float(delta.max()),
        "z_shape": list(features["z_rl"].shape),
        "reference_shape": list(features["ref_chunk"].shape),
        "scope": "frozen VLA reference; not the trained RLT student",
    }
    (d / "reference-eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
