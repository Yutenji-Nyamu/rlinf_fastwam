"""CPU contract checks for calibration-only gates and paired workload."""

import unittest

import numpy as np
from protocol import choose
from threshold_sweep import conditions


class ThresholdSweepTests(unittest.TestCase):
    def test_gates_cover_both_tails_at_requested_calibration_rates(self):
        # Known quantiles independently check the gate direction and strict ties.
        values = np.linspace(0, 1, 1001)
        rows = conditions([{"u": value} for value in values])
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({row["condition"] for row in rows}), 30)
        for row in rows:
            gate = {row["policy"]: row["threshold"]}
            selected = choose(row["policy"], values, np.ones(1001, bool), gate, None)
            self.assertAlmostEqual(selected.mean(), row["target_fraction"], delta=0.002)
        self.assertEqual({r["policy"] for r in rows}, {"high", "low"})

    def test_calibration_must_be_finite(self):
        for values in ([], [{"u": float("nan")}], [{"u": float("inf")}]):
            with self.assertRaises(AssertionError):
                conditions(values)


if __name__ == "__main__":
    unittest.main()
