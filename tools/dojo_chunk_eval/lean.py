# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Native success receipts, no video/signals; bounded observation timing probe."""

import os
import time
from pathlib import Path

from common import progress, read, save
from layouts import install_plan


def instrument(env):
    cfg = read(os.environ["DOJO_U_CONFIG"])
    out = Path(cfg["output"])
    install_plan(env, cfg["layout_plan"])
    native_obs, native_action, native_reset = (
        env.get_obs_batch,
        env.take_action_batch,
        env.reset,
    )
    last = [0.0]
    timings = {
        "observation_seconds": 0.0,
        "action_seconds": 0.0,
        "observation_calls": 0,
    }
    env._stream_vision = lambda *args, **kwargs: None

    def obs(*args, **kwargs):
        t = time.monotonic()
        result = native_obs(*args, **kwargs)
        timings["observation_seconds"] += time.monotonic() - t
        timings["observation_calls"] += 1
        for row in result:
            i = int(row["env_idx"])
            row["_eval_meta"] = {
                "layout": int(env.env_seeds[i]),
                "query": int(env.take_action_cnt[i]) // cfg["execute_chunk"],
            }
        return result

    def action(*args, **kwargs):
        t = time.monotonic()
        result = native_action(*args, **kwargs)
        timings["action_seconds"] += time.monotonic() - t
        if time.monotonic() - last[0] > 5:
            progress("action", counts=env.take_action_cnt, **timings)
            last[0] = time.monotonic()
        return result

    def reset(*args, **kwargs):
        result = native_reset(*args, **kwargs)
        progress("reset_complete", layouts=env.current_env_seed_map)
        return result

    def complete(i, video_path, tag):
        layout = int(env.env_seeds[i])
        save(
            out / "episodes" / f"layout_{layout:05d}.json",
            dict(
                task=cfg["task"],
                layout=layout,
                success=bool(env.success[i]),
                native_tag=tag,
                actions=int(env.take_action_cnt[i]),
                source_layout=cfg["layout_plan"][layout],
                num_steps=cfg["num_steps"],
                chunk=cfg["execute_chunk"],
                probe_only=bool(cfg.get("probe_queries")),
                **timings,
            ),
        )

    env.get_obs_batch, env.take_action_batch, env.reset, env.save_video = (
        obs,
        action,
        reset,
        complete,
    )
    save(
        out / "native-path.json",
        {"result": str(Path(env.save_dir).resolve() / "_result.json")},
    )
    return env
