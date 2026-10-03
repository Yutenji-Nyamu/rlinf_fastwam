"""Targeted CPU checks; run in the isolated server RLinf checkout, not Windows."""

import unittest

import numpy as np

from rlinf.envs.world_model.opendw_adapter import (
    compose_robotwin_views,
    map_frame_scores,
    next_command_state,
    prepare_raw_actions,
    split_robotwin_views,
)


class AdapterTests(unittest.TestCase):
    def test_raw_state_and_gripper_without_mutating_policy_action(self):
        raw = np.arange(32 * 14, dtype=np.float32).reshape(1, 32, 14) / 100
        raw[..., 6] = -0.4
        raw[..., 13] = 1.5
        prepared = prepare_raw_actions(raw)
        np.testing.assert_equal(raw[..., 6], -0.4)
        np.testing.assert_equal(prepared[..., 6], 0)
        np.testing.assert_equal(prepared[..., 13], 1)
        state = next_command_state(prepared)
        np.testing.assert_equal(state[:, :6], raw[:, -1, :6])
        np.testing.assert_equal(state[:, 7:13], raw[:, -1, 7:13])

    def test_views_have_stable_identity(self):
        views = [np.full((256, 256, 3), index * 80, dtype=np.uint8) for index in range(3)]
        canvas = compose_robotwin_views(*views)
        self.assertEqual(canvas.shape, (384, 320, 3))
        for expected, actual in zip(views, split_robotwin_views(canvas)):
            np.testing.assert_equal(expected, actual)

    def test_reward_telescope_chunk_done_and_true_frame_location(self):
        scores = np.array([[0.2, 0.3, 0.95, 0.7, 0.6, 0.5, 0.4, 0.3]], dtype=np.float32)
        result = map_frame_scores(scores, [0.1], [0], max_episode_steps=64, reward_coef=5)
        self.assertAlmostEqual(float(result.rewards.sum()), 1.0, places=5)
        self.assertEqual(result.first_success_action.tolist(), [12])
        self.assertEqual(np.flatnonzero(result.terminated[0]).tolist(), [31])
        self.assertFalse(result.truncated.any())
        self.assertEqual(np.flatnonzero(result.rewards[0]).tolist(), [3, 7, 11, 15, 19, 23, 27, 31])

    def test_budget_is_actions_and_partial_chunks_fail_explicitly(self):
        scores = np.zeros((1, 8), dtype=np.float32)
        result = map_frame_scores(scores, [0], [0], max_episode_steps=32)
        self.assertTrue(result.truncated[0, 31])
        self.assertFalse(result.terminated.any())
        with self.assertRaises(ValueError):
            map_frame_scores(scores, [0], [384], max_episode_steps=400)

    def test_invalid_actions_or_scores_fail_before_inference(self):
        with self.assertRaises(ValueError):
            prepare_raw_actions(np.zeros((1, 50, 14)))
        with self.assertRaises(ValueError):
            map_frame_scores(np.full((1, 8), np.nan), [0], [0], max_episode_steps=32)


if __name__ == "__main__":
    unittest.main()
