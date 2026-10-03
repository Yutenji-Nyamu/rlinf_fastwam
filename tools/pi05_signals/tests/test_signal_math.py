"""NumPy-only numerical contract tests; run on the designated server.

Example: python -m unittest discover -s signal_inference_20261003/tests
"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import signal_math as sm


class SignalMathTests(unittest.TestCase):
    def assert_score(self, result, expected, valid=True):
        np.testing.assert_allclose(result.score, expected, rtol=1e-7, atol=1e-8)
        np.testing.assert_array_equal(result.valid, np.broadcast_to(valid, result.score.shape))

    def test_population_vector_variance_is_not_flattened_variance(self):
        t = np.arange(7.0)
        vector = np.stack((t + 100.0, 2.0 * t - 100.0), axis=-1)
        x = np.broadcast_to(vector[None, :, None, :], (2, 7, 3, 2)).copy()
        x += np.arange(2.0)[:, None, None, None]
        x += np.arange(3.0)[None, None, :, None]
        tail = x[:, -5:]
        expected = np.var(tail, axis=1, ddof=0).sum(axis=-1)
        direct = np.mean(np.sum((tail - tail.mean(axis=1, keepdims=True)) ** 2, axis=-1), axis=1)
        self.assert_score(sm.dv_score(x), np.full((2, 3), 10.0))
        self.assert_score(sm.fresco_score(x), expected)
        np.testing.assert_allclose(expected, direct)
        self.assertFalse(np.allclose(expected, np.var(tail, axis=(1, 3), ddof=0)))

    def test_tail_uses_last_five_forward_evaluations(self):
        x = np.zeros((2, 10, 3, 1))
        x[:, :5] = np.arange(5.0)[None, :, None, None] * 100.0
        self.assert_score(sm.dv_score(x), np.zeros((2, 3)))
        x[:, 9] = 2.0
        self.assert_score(sm.dv_score(x), np.full((2, 3), 0.64))
        self.assert_score(sm.fresco_score(x), np.full((2, 3), 0.64))
        with self.assertRaises(ValueError):
            sm.dv_score(x[:, :4])

    def test_variance_nonfinite_mask_only_in_selected_window_and_action(self):
        x = np.zeros((2, 7, 3, 2))
        x[0, 0, 0, 0] = np.nan  # Outside the tail window.
        x[1, 6, 2, 1] = np.inf
        result = sm.dv_score(x)
        expected_mask = np.ones((2, 3), dtype=bool)
        expected_mask[1, 2] = False
        np.testing.assert_array_equal(result.valid, expected_mask)
        self.assertTrue(np.isnan(result.score[1, 2]))
        np.testing.assert_allclose(result.score[result.valid], 0.0)

    def test_geo_is_order_sensitive_and_preserves_batch_and_action(self):
        velocity = np.zeros((2, 3, 4, 1))
        velocity[0, :, :, 0] = np.array([1.0, 2.0, 3.0])[:, None]
        velocity[1, :, :, 0] = np.array([1.0, 3.0, 2.0])[:, None]
        result = sm.geo_full(velocity)
        self.assert_score(result, np.array([[1.0] * 4, [1.5] * 4]))
        np.testing.assert_allclose(np.var(velocity[0], axis=0), np.var(velocity[1], axis=0))
        self.assert_score(sm.geo_full(np.zeros_like(velocity)), np.zeros((2, 4)))

    def test_geo_invalid_is_local_to_action(self):
        velocity = np.ones((2, 3, 4, 2))
        velocity[0, 1, 2, 0] = np.nan
        result = sm.geo_full(velocity)
        self.assertFalse(result.valid[0, 2])
        self.assertTrue(np.isnan(result.score[0, 2]))
        self.assertEqual(int(result.valid.sum()), 7)

    def test_geoaac_falling_prefix_is_zero_positive_growth(self):
        # Action 1 changes velocity 1 -> 2. Appended action 2 is a constant
        # large velocity 10 -> 10, so the normalized prefix score falls.
        velocity = np.array([[[[1.0], [10.0]], [[2.0], [10.0]]]])
        result = sm.geoaac_growth(velocity, np.array([1.0, 0.5]))
        self.assertTrue(np.isnan(result.score[0, 0]))
        self.assertFalse(result.valid[0, 0])
        self.assertTrue(result.valid[0, 1])
        self.assertEqual(result.score[0, 1], 0.0)

    def test_geoaac_stage_weights_and_prefix_formula(self):
        velocity = np.array([
            [[[1.0], [1.0], [2.0]], [[1.1], [2.0], [1.0]], [[1.2], [4.0], [5.0]]],
            [[[2.0], [1.0], [2.0]], [[2.1], [2.0], [3.0]], [[2.2], [5.0], [4.0]]],
        ])
        times = np.array([1.0, 0.7, 0.1])
        eps = 1e-8
        midpoint = (np.arange(2) + 0.5) / 3
        weights = midpoint * (1 - midpoint) ** 2
        weights /= weights.mean()
        expected = np.full((2, 3), np.nan)
        for batch in range(2):
            prefixes = []
            for h in range(1, 4):
                prefix = velocity[batch, :, :h, :]
                speed = sum(np.linalg.norm(row.ravel()) for row in prefix)
                changes = [np.linalg.norm(((prefix[j + 1] - prefix[j]) /
                           (abs(times[j + 1] - times[j]) + eps)).ravel()) for j in range(2)]
                prefixes.append(3 * np.dot(weights, changes) / (speed + eps))
            for h in range(1, 3):
                expected[batch, h] = max((prefixes[h] - prefixes[h - 1]) /
                                         (abs(prefixes[h - 1]) + eps), 0.0)
        result = sm.geoaac_growth(velocity, times)
        np.testing.assert_allclose(result.score, expected, equal_nan=True)
        np.testing.assert_array_equal(result.valid, [[False, True, True], [False, True, True]])
        reverse_time = 1.0 - times
        np.testing.assert_allclose(sm.geoaac_growth(velocity, reverse_time).score,
                                   result.score, equal_nan=True)

    def test_geoaac_nonfinite_prefix_propagates_and_single_action_is_missing(self):
        velocity = np.ones((2, 3, 4, 1))
        velocity[0, 1, 2, 0] = np.nan
        result = sm.geoaac_growth(velocity, np.array([1.0, 0.5, 0.0]))
        np.testing.assert_array_equal(result.valid, [[False, True, False, False],
                                                   [False, True, True, True]])
        one = sm.geoaac_growth(np.ones((2, 3, 1, 1)), np.array([1.0, 0.5, 0.0]))
        np.testing.assert_array_equal(one.valid, [[False], [False]])
        self.assertTrue(np.isnan(one.score).all())

    def test_shift_log_distance_and_local_mask(self):
        first = np.zeros((2, 3, 2))
        last = np.broadcast_to([3.0, 4.0], first.shape).copy()
        self.assert_score(sm.shift_score(first, last), np.full((2, 3), np.log1p(5.0)))
        last[1, 2, 0] = np.nan
        result = sm.shift_score(first, last)
        self.assertEqual(int(result.valid.sum()), 5)
        self.assertTrue(np.isnan(result.score[1, 2]))

    def test_norm_uses_tail_five_deepest_three_and_excludes_others(self):
        norms = np.full((2, 8, 5, 4), 999.0)
        norms[:, -5:, -3:, :] = np.arange(12.0).reshape(1, 1, 3, 4)
        norms[0, 0, 0, 0] = np.nan
        expected = np.broadcast_to(np.arange(12.0).reshape(3, 4).mean(axis=0), (2, 4))
        self.assert_score(sm.norm_score(norms), expected)
        norms[1, -1, -1, 2] = -1.0
        result = sm.norm_score(norms)
        self.assertFalse(result.valid[1, 2])
        self.assertTrue(np.isnan(result.score[1, 2]))

    def test_sr_rank_one_and_two_without_centering(self):
        hidden = np.zeros((2, 5, 2))
        hidden[0] = np.array([1.0, 1.0])  # Nonzero repeated rows: stable rank 1.
        hidden[1, 0] = [1.0, 0.0]
        hidden[1, 1] = [0.0, 1.0]  # Equal singular values: stable rank 2.
        self.assert_score(sm.sr_window5(hidden), np.array([[1.0] * 5, [2.0] * 5]))
        self.assert_score(sm.sr_window5(hidden * 1e-100), np.array([[1.0] * 5, [2.0] * 5]))

    def test_sr_does_not_unit_normalize_rows(self):
        hidden = np.zeros((1, 5, 2))
        hidden[0, 0] = [2.0, 0.0]
        hidden[0, 1] = [0.0, 1.0]
        self.assert_score(sm.sr_window5(hidden), np.full((1, 5), 1.25))

    def test_sr_fixed_window_boundary_shift_and_invalid_window(self):
        hidden = np.zeros((2, 7, 2))
        hidden[:, :, 0] = 1.0
        hidden[:, 6] = [0.0, 1.0]
        result = sm.sr_window5(hidden)
        self.assert_score(result, np.array([[1, 1, 1, 1, 1.25, 1.25, 1.25]] * 2))
        hidden[0, 0, 0] = np.nan
        invalid = sm.sr_window5(hidden)
        np.testing.assert_array_equal(invalid.valid[0], [False, False, False, True, True, True, True])
        zero = sm.sr_window5(np.zeros((2, 5, 2)))
        self.assertFalse(zero.valid.any())
        self.assertTrue(np.isnan(zero.score).all())

    def test_ugrow_identical_zero_and_opposite_sign(self):
        a = np.ones((2, 3, 4))
        self.assert_score(sm.ugrow_10_vs_5(a, a), np.zeros((2, 3)))
        self.assert_score(sm.ugrow_10_vs_5(a * 0, a * 0), np.zeros((2, 3)))
        self.assert_score(sm.ugrow_10_vs_5(a, -a), np.full((2, 3), 1.0 / (1.0 + 1e-8)))

    def test_ugrow_coordinate_ratio_then_mean_and_sqrt_two_factor(self):
        a = np.broadcast_to([3.0, 0.0], (2, 3, 2)).copy()
        b = np.broadcast_to([1.0, 2.0], (2, 3, 2)).copy()
        coordinate_expected = np.array([1.0 / (np.sqrt(5.0) + 1e-8),
                                        1.0 / (np.sqrt(2.0) + 1e-8)])
        self.assert_score(sm.ugrow_10_vs_5(a, b), np.full((2, 3), coordinate_expected.mean()))
        b[0, 2, 0] = np.inf
        result = sm.ugrow_10_vs_5(a, b)
        self.assertFalse(result.valid[0, 2])
        self.assertTrue(np.isnan(result.score[0, 2]))

    def test_malformed_shapes_and_parameters_fail_explicitly(self):
        with self.assertRaises(ValueError):
            sm.dv_score(np.zeros((5, 3, 2)))
        with self.assertRaises(ValueError):
            sm.dv_score(np.zeros((1, 5, 2, 1)), tail_n=0)
        with self.assertRaises(ValueError):
            sm.norm_score(np.ones((1, 5, 2, 4)))
        with self.assertRaises(ValueError):
            sm.sr_window5(np.ones((1, 4, 3)))
        with self.assertRaises(ValueError):
            sm.shift_score(np.zeros((2, 3, 4)), np.zeros((1, 3, 4)))
        with self.assertRaises(ValueError):
            sm.ugrow_10_vs_5(np.zeros((2, 3, 4)), np.zeros((2, 3, 1)))
        with self.assertRaises(ValueError):
            sm.geo_full(np.ones((1, 2, 3, 1)), epsilon=0)
        with self.assertRaises(ValueError):
            sm.geoaac_growth(np.ones((1, 3, 2, 1)), np.array([1.0, 0.5]))
        for bad_times in ([1.0, 1.0, 0.0], [1.0, 0.0, 0.5], [1.0, np.nan, 0.0]):
            with self.assertRaises(ValueError):
                sm.geoaac_growth(np.ones((1, 3, 2, 1)), np.array(bad_times))


if __name__ == "__main__":
    unittest.main()
