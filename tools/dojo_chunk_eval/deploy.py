# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Native Pi_05 batch loop with explicit prefix and optional redundant-observation removal."""

import os
from pathlib import Path

from common import read, save


def eval_one_episode_batch(TASK_ENV, model_client):
    cfg = read(os.environ["DOJO_U_CONFIG"])
    model_client.call(func_name="reset")
    query = 0
    while not TASK_ENV.is_episode_end():
        ids = TASK_ENV.get_running_env_idx_list()
        obs = TASK_ENV.get_obs_batch(ids)
        model_client.call(func_name="update_obs_batch", obs=obs)
        actions = model_client.call(func_name="get_action_batch", obs=ids)
        chunk = min(len(actions[0]), cfg["execute_chunk"])
        assert len(actions[0]) == 50 and chunk in (10, 50)
        for action_idx in range(chunk):
            TASK_ENV.take_action_batch([a[action_idx] for a in actions], ids)
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk:
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [i for i, j in enumerate(ids) if j in running]
            actions = [actions[i] for i in active]
            ids = [ids[i] for i in active]
            if not cfg.get("decision_observations_only", False):
                model_client.call(
                    func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(ids)
                )
            elif TASK_ENV.physx_monitor_enabled:
                # These checks formerly also ran inside get_obs_batch.
                TASK_ENV._check_physx_broken_envs()
                TASK_ENV._check_endpose_finite(ids)
        query += 1
        if cfg.get("probe_queries") and query >= cfg["probe_queries"]:
            save(
                Path(cfg["output"]) / "probe-complete.json",
                {"queries": query, "counts": TASK_ENV.take_action_cnt},
            )
            break


def eval_one_episode(TASK_ENV, model_client):
    return eval_one_episode_batch(TASK_ENV, model_client)
