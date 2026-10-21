# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Verify the real service keeps history aligned across B16 and a short tail."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "wovr_cycle_20261008/code"))
from opendw_service_batched import Backend, batch_ranges


class BatchHistoryTests(unittest.TestCase):
    def test_b16_and_tail_keep_video_history_alignment(self):
        calls = []

        def score(future, history, lengths, instructions):
            calls.append(
                (
                    future[:, 0, 0, 0, 0].tolist(),
                    history[:, 0, 0, 0, 0].tolist(),
                    lengths.tolist(),
                    instructions,
                )
            )
            return (future[:, 0, 0, 0, 0] / 100).tolist()

        backend = object.__new__(Backend)
        backend.args = SimpleNamespace(reward_backend="topreward")
        backend.reward = SimpleNamespace(score_windows=score)
        history = np.broadcast_to(
            np.arange(17, dtype=np.uint8)[:, None, None, None, None],
            (17, 8, 256, 320, 3),
        )
        data = {
            "reward_history": history,
            "reward_history_lengths": np.full(17, 8, np.int64),
        }
        scores = []
        for start, end in batch_ranges(17, 16):
            videos = [
                [Image.new("RGB", (320, 384), (row + 20,) * 3)] * 9
                for row in range(start, end)
            ]
            scores.extend(
                backend._score_videos(
                    videos, [str(i) for i in range(start, end)], data, start
                )
            )
        self.assertEqual([len(c[0]) for c in calls], [16, 1])
        self.assertEqual(calls[-1], ([36], [16], [8], ["16"]))
        self.assertEqual(calls[0][1], list(range(16)))
        np.testing.assert_allclose(
            np.stack(scores), np.repeat((np.arange(17) + 20)[:, None] / 100, 8, axis=1)
        )


if __name__ == "__main__":
    unittest.main()
