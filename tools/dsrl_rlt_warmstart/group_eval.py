# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""One compact zero-training screen of temporal noise groups; no replay updates."""

import copy
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main(directory):
    d = Path(directory)
    p = json.loads((d / "plan.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p["gpu"])
    from rlinf.models.embodiment.openpi import get_model

    if p["bench"] == "dojo":
        from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv as Environment
    else:
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv as Environment
    c = OmegaConf.load(d / "config/formal.yaml")
    torch.manual_seed(1234)
    np.random.seed(1234)
    model = get_model(c.actor.model).eval().cuda()
    assert int(model.dsrl_policy_phase) == 0 and not model.config.dsrl_u_enabled
    ec = copy.deepcopy(c.env.eval)
    ec.total_num_envs = 8
    ec.auto_reset = False
    ec.ignore_terminations = False
    env = Environment(ec, 8, 0, 1, None)
    schedules = []
    cases = {}
    chunk = int(c.actor.model.num_action_chunks)
    horizon = int(ec.max_episode_steps)
    for groups in (50, 10, 5):
        # No learned-policy forward is used here. Only the Gaussian sampler reads
        # these fields; a fresh formal process builds the selected compact actor.
        model.config = dataclasses.replace(
            model.config,
            dsrl_noise_groups=groups,
            dsrl_action_noise_dim=groups * model.config.action_dim,
        )
        rows = []
        for wave in range(2):
            if groups == 50:
                if wave:
                    env.update_reset_state_ids()
                schedules.append(env.reset_state_ids.tolist())
            started = time.time()
            env.reset_state_ids = torch.tensor(schedules[wave], dtype=torch.long)
            obs, _ = env.reset(env_seeds=schedules[wave])
            ids = env.reset_state_ids.tolist()
            success = torch.zeros(8, dtype=torch.bool)
            done = success.clone()
            for step in range((horizon + chunk - 1) // chunk):
                o = dict(obs)
                o.setdefault("extra_view_images", None)
                with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(96000 + 100 * wave + step)
                    torch.cuda.manual_seed(96000 + 100 * wave + step)
                    actions, _ = model.predict_action_batch(o, mode="eval")
                assert np.isfinite(
                    np.asarray(
                        actions.detach().cpu()
                        if isinstance(actions, torch.Tensor)
                        else actions
                    )
                ).all()
                nxt, _, term, trunc, _ = env.chunk_step(actions)
                term = torch.as_tensor(term).bool().reshape(8, -1).any(-1).cpu()
                trunc = torch.as_tensor(trunc).bool().reshape(8, -1).any(-1).cpu()
                success |= term & ~done
                done |= term | trunc
                obs = nxt[-1]
                if done.all():
                    break
            row = {
                "wave": wave,
                "requested_ids": schedules[wave],
                "actual_ids": ids,
                "success": success.tolist(),
                "seconds": time.time() - started,
            }
            rows.append(row)
            print(json.dumps({"groups": groups, **row}), flush=True)
        cases[str(groups)] = {
            "successes": sum(sum(x["success"]) for x in rows),
            "episodes": 16,
            "waves": rows,
        }
        save(d / "group-eval-progress.json", {"cases": cases, "time": time.time()})
    base = cases["50"]["successes"]
    # A positive baseline is necessary to interpret an all-zero screen. Prefer
    # the lowest dimension which has not lost successes in this small sample.
    selected = next(
        (g for g in (5, 10) if base > 0 and cases[str(g)]["successes"] >= base), 50
    )
    for f in (d / "config").glob("*.yaml"):
        cfg = OmegaConf.load(f)
        for model_cfg in (cfg.actor.model, cfg.rollout.model):
            model_cfg.openpi.dsrl_noise_groups = selected
            model_cfg.openpi.dsrl_action_noise_dim = selected * 32
        cfg.algorithm.entropy_tuning.target_entropy = -selected * 32 / 2
        OmegaConf.save(cfg, f)
    save(
        d / "group-eval.json",
        {
            "time": time.time(),
            "cases": cases,
            "selected_groups": selected,
            "scope": "small zero-training screen; not a statistical equivalence claim",
        },
    )
    # SAPIEN has previously crashed in interpreter shutdown after completed work.
    # Release normal environment resources, then bypass global destructor order.
    env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
