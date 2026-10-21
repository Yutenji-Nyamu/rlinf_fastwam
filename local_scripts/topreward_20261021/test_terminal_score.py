# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Terminal probability must survive the actual embodied GRPO aggregation."""

import unittest

import numpy as np
import torch

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.envs.world_model.opendw_adapter import map_frame_scores


class TerminalScoreTests(unittest.TestCase):
    def test_fixed_budget_scores_reach_group_relative_advantages(self):
        # Two reset groups: one has a useful ordering, one has no signal.
        final = np.array(
            [0.1, 0.3, 0.2, 0.8, 0.4, 0.6, 0.9, 0.5] + [0.4] * 8, dtype=np.float32
        )
        rewards = []
        dones = [torch.zeros((16, 32), dtype=torch.bool)]
        for chunk in range(12):
            score = np.repeat((final if chunk == 11 else 1 - final)[:, None], 8, 1)
            result = map_frame_scores(
                score,
                np.zeros(16),
                np.full(16, chunk * 32),
                max_episode_steps=384,
                relative_reward=False,
                reward_mode="terminal_score",
                success_threshold=0.01,
            )
            self.assertFalse(result.terminated.any())
            self.assertTrue((result.first_success_action == -1).all())
            self.assertEqual(bool(result.truncated.any()), chunk == 11)
            rewards.append(torch.from_numpy(result.rewards))
            dones.append(torch.from_numpy(result.terminated | result.truncated))
        rewards = torch.stack(rewards)
        torch.testing.assert_close(rewards.sum((0, 2)), torch.from_numpy(final))
        self.assertFalse(rewards[:-1].any())
        actual = calculate_adv_and_returns(
            task_type="embodied",
            adv_type="grpo",
            rewards=rewards,
            dones=torch.stack(dones),
            values=None,
            group_size=8,
            num_action_chunks=32,
            reward_type="chunk_level",
            loss_mask=torch.ones((12, 16, 1), dtype=torch.bool),
            loss_mask_sum=None,
        )["advantages"]
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual[0], actual[-1])
        self.assertEqual(
            actual[0, :8, 0].argsort().tolist(), final[:8].argsort().tolist()
        )
        self.assertLess(actual[:, 8:].abs().max().item(), 0.1)
        self.assertLess(actual[:, :8].mean().abs().item(), 1e-5)

    def test_rejects_shaping_and_partial_budget(self):
        args = (np.full((1, 8), 0.8), np.zeros(1), np.zeros(1))
        with self.assertRaises(ValueError):
            map_frame_scores(
                *args,
                max_episode_steps=384,
                reward_mode="terminal_score",
                relative_reward=True,
            )
        with self.assertRaises(ValueError):
            map_frame_scores(
                *args,
                max_episode_steps=16,
                reward_mode="terminal_score",
                relative_reward=False,
            )


if __name__ == "__main__":
    unittest.main()
