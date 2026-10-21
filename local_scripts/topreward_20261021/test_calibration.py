# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Focused CPU tests for early alarms, split leakage and explicit clip windows."""

import unittest

import numpy as np
from calibration import metrics, report, select_threshold
from qwen_reward import window_frames


def row(seed, probability, success, split="calibration"):
    return {
        "actual_seed": seed,
        "probability": probability,
        "native_success_once": success,
        "split": split,
        "old_rm_probability": probability,
        "legacy_rm_split": "test",
    }


class CalibrationTests(unittest.TestCase):
    def test_successful_trajectory_can_still_have_early_alarm(self):
        m = metrics([row(1, 0.8, False), row(1, 0.9, True), row(2, 0.1, False)], 0.7)
        self.assertEqual(m["early_alarm_episodes"], 1)
        self.assertEqual(m["success_recall"], 1.0)
        self.assertEqual(m["episodes"], 2)

    def test_separable_threshold_preserves_detection(self):
        r = select_threshold(
            [row(1, 0.8, False), row(1, 0.9, True), row(2, 0.2, False)]
        )
        self.assertTrue(r["feasible"])
        self.assertGreater(r["threshold"], 0.8)
        self.assertLessEqual(r["threshold"], float(np.float32(0.9)))

    def test_mixed_scores_do_not_produce_fake_threshold_above_one(self):
        r = select_threshold([row(1, 1.0, False), row(1, 0.9, True)])
        self.assertFalse(r["feasible"])
        self.assertIsNone(r["threshold"])

    def test_holdout_cannot_be_used_for_selection(self):
        with self.assertRaises(ValueError):
            select_threshold([row(1, 0.9, True, "holdout")])
        with self.assertRaises(ValueError):
            report([row(1, 0.2, False), row(1, 0.9, True, "holdout")], 0.5)

    def test_threshold_uses_wire_float32_precision(self):
        p = float(np.float32(0.8))
        r = select_threshold([row(1, p, False), row(1, 1.0, True)])
        self.assertTrue(r["feasible"])
        self.assertEqual(
            metrics([row(1, p, False)], r["threshold"])["early_alarm_episodes"], 0
        )

    def test_windows_have_no_cross_slot_or_reset_state(self):
        def frames(values):
            return np.asarray(values, dtype=np.uint8)[:, None, None, None].repeat(
                3, axis=3
            )

        a = window_frames(frames(range(8)), frames(range(8, 16)))
        b = window_frames(frames(range(100, 108)), frames(range(108, 116)))
        reset = window_frames(frames([]), frames(range(8)))
        self.assertEqual(a[:, 0, 0, 0].tolist(), list(range(16)))
        self.assertEqual(b[:, 0, 0, 0].tolist(), list(range(100, 116)))
        self.assertEqual(reset[:, 0, 0, 0].tolist(), list(range(8)))
        self.assertEqual(
            window_frames(a, frames([16, 17]))[:, 0, 0, 0].tolist(), list(range(2, 18))
        )


if __name__ == "__main__":
    unittest.main()
