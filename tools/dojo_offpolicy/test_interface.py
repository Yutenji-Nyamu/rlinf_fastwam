# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""The changed boundary: native once-only success -> compact DSRL transitions."""

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import project_dsrl_trajectory
from rlinf.envs.robodojo.protocol import terminal_arrays
from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv


def test_fixed_evaluation_covers_32_layouts_in_four_waves_and_repeats():
    env = RoboDojoEnv.__new__(RoboDojoEnv)
    env.cfg = OmegaConf.create({"use_fixed_reset_state_ids": True})
    env.num_envs, env.group_size, env.pool_size = 8, 1, 100
    env._fixed_offset, env._fixed_cycle = 0, 32
    env._rng = np.random.RandomState(42)
    for _ in range(2):
        layouts = []
        for _ in range(4):
            env.update_reset_state_ids()
            layouts.extend(env.reset_state_ids.tolist())
        assert layouts == list(range(32))


def test_native_success_timeout_and_postterminal_projection():
    done = np.zeros((2, 50), bool)
    success = np.zeros_like(done)
    done[0, 16:] = success[0, 16:] = True
    r0, t0, x0 = terminal_arrays(np.array([False, False]), done, success)
    done[0, :] = success[0, :] = True
    done[1, -1] = True
    r1, t1, x1 = terminal_arrays(np.array([True, False]), done, success)
    assert r0.sum() == 1 and r1.sum() == 0
    terms = torch.from_numpy(np.stack([t0, t1]))
    truncs = torch.from_numpy(np.stack([x0, x1]))
    images = torch.zeros((2, 2, 64, 64, 3), dtype=torch.uint8)
    states = torch.arange(56).float().reshape(2, 2, 14)
    trajectory = Trajectory(
        actions=torch.zeros((2, 2, 50, 32)),
        rewards=torch.from_numpy(np.stack([r0, r1])),
        terminations=terms,
        truncations=truncs,
        dones=terms | truncs,
        curr_obs={"main_images": images, "states": states},
        next_obs={"main_images": images, "states": states + 1},
    )
    batch = project_dsrl_trajectory(
        trajectory,
        action_horizon=50,
        latent_dim=32,
        state_dim=14,
        num_action_chunks=50,
        gamma=0.999,
    )
    assert batch["actions"].shape == (3, 32)
    assert batch["rewards"].flatten().tolist() == [0.0, -1.0, -1.0]
    assert batch["continuations"].flatten().tolist() == [False, True, True]
    assert batch["truncations"].flatten().tolist() == [False, False, True]
    assert torch.equal(
        batch["next_obs"]["states"],
        torch.stack([states[0, 0] + 1, states[0, 1] + 1, states[1, 1] + 1]),
    )
    assert torch.allclose(batch["discounts"], torch.full((3, 1), 0.999**50))
