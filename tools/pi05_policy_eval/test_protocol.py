# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks for new intervention semantics, run on the server."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from protocol import choose, save, thresholds, validate_config


class ProtocolTests(unittest.TestCase):
    def test_equal_and_inactive_states_do_not_trigger_threshold_gate(self):
        active = np.array([True, True, False])
        u = np.array([0.2, 0.8, 0.9])
        self.assertEqual(
            choose("high", u, active, {"high": 0.2}, None).tolist(),
            [False, True, False],
        )
        self.assertEqual(
            choose("low", u, active, {"low": 0.8}, None).tolist(), [True, False, False]
        )
        self.assertEqual(
            choose("random", u, active, {}, np.array([0.1, 0.9, 0])).tolist(),
            [True, False, False],
        )

    def test_thresholds_only_read_declared_calibration_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calibration.json"
            save(p, [{"u": x} for x in range(7)])
            save(Path(d) / "test.json", [{"u": 1000}])
            result = thresholds([p])
            self.assertEqual((result["low"], result["high"]), (2.0, 4.0))

    def test_execution_cannot_exceed_prediction(self):
        c = {
            "num_envs": 32,
            "step_limit": 200,
            "num_steps": 10,
            "execute_chunk": 100,
            "model": {"openpi": {"action_chunk": 50}},
            "policy": "weak",
        }
        with self.assertRaises(AssertionError):
            validate_config(c)


if __name__ == "__main__":
    unittest.main()
