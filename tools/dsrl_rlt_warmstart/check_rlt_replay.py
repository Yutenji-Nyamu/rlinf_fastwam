# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Check nonterminal tails and first-done replay alignment without a simulator."""

import json

import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy

worker = object.__new__(RLTACFSDPPolicy)
worker.cfg = OmegaConf.create(
    {
        "algorithm": {
            "rlt_transition_replay": {
                "compact": True,
                "bootstrap_on_truncation": True,
            },
            "dv_observe": False,
        },
        "env": {"train": {"auto_reset": False}},
    }
)
worker.replay_buffer = object.__new__(TrajectoryReplayBuffer)
trajectory = Trajectory(max_episode_length=2)
trajectory.actions = torch.arange(3 * 2 * 2 * 14).reshape(3, 2, 2, 14).float()
trajectory.rewards = torch.arange(8).reshape(2, 2, 2).float()
trajectory.dones = torch.zeros(3, 2, 2, dtype=torch.bool)
trajectory.terminations = torch.zeros_like(trajectory.dones)
trajectory.truncations = torch.zeros_like(trajectory.dones)
trajectory.forward_inputs = {"record_transition": torch.ones(3, 2, dtype=torch.bool)}
trajectory.curr_obs = {
    "z_rl": torch.arange(12).reshape(2, 2, 3).float(),
    "proprio": torch.zeros(2, 2, 14),
    "ref_chunk": torch.zeros(2, 2, 2, 14),
}
trajectory.next_obs = {key: value + 1 for key, value in trajectory.curr_obs.items()}
rows, completed = worker._transition_replay_trajectories(trajectory)
assert len(rows) == 4 and completed == 0
for row, (env, step) in zip(rows, [(0, 0), (0, 1), (1, 0), (1, 1)]):
    assert torch.equal(row.actions[0, 0], trajectory.actions[step, env])
    assert torch.equal(row.rewards[0, 0], trajectory.rewards[step, env])
    assert torch.equal(
        row.next_obs["z_rl"][0, 0], trajectory.next_obs["z_rl"][step, env]
    )
trajectory.dones[1, 0, 0] = True
trajectory.terminations[1, 0, 0] = True
trajectory.dones[2, 1, 1] = True
trajectory.truncations[2, 1, 1] = True
rows, completed = worker._transition_replay_trajectories(trajectory)
assert len(rows) == 3 and completed == 2
assert torch.equal(rows[0].next_obs["z_rl"], rows[0].curr_obs["z_rl"])
assert torch.equal(rows[-1].next_obs["z_rl"][0, 0], trajectory.next_obs["z_rl"][1, 1])
print(
    json.dumps(
        {"nonterminal_tail_rows": 4, "first_done_rows": 3, "bootstrap_truncation": True}
    )
)
