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
from protocol import choose, configure_recording_wait, execution_lengths, save, thresholds, validate_config, variable_chunk_step


class ProtocolTests(unittest.TestCase):
    def test_native_vector_receives_whole_mixed_prefixes_and_exact_remaining_budget(self):
        # Execute the installed VectorEnv.step against lightweight SubEnv doubles.
        # This exercises its dispatch contract without constructing CUDA/SAPIEN.
        import ast
        import os
        from concurrent.futures import ThreadPoolExecutor
        from types import MethodType, SimpleNamespace
        source = Path(os.environ["ROBOTWIN_PATH"]) / "robotwin/envs/vector_env.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VectorEnv")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("step", "transform")]
        from collections import defaultdict
        namespace = {"defaultdict": defaultdict}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        received = []

        class Slot:
            def __init__(self, index): self.index = index
            def step(self, actions):
                received.append((self.index, actions.copy()))
                return dict(obs=self.index, reward=0, terminated=False, truncated=False, info={})

        with ThreadPoolExecutor(4) as pool:
            native = SimpleNamespace(envs=[Slot(i) for i in range(4)], env_thread_pool=pool, n_envs=4)
            native.step = MethodType(namespace["step"], native)
            native.transform = lambda results: namespace["transform"](native, results)
            env = SimpleNamespace(venv=native, _extract_obs_image=lambda observations: observations)
            actions = np.arange(4 * 50 * 14).reshape(4, 50, 14)
            lengths = execution_lengths(np.array([True, False, True, False]),
                                        np.array([True, True, True, False]),
                                        np.array([0, 0, 190, 200]), mode="c20")
            self.assertEqual(lengths.tolist(), [20, 50, 10, 0])
            variable_chunk_step(env, actions, lengths)
            configure_recording_wait(env)
            variable_chunk_step(env, actions, lengths)
            self.assertEqual(len(received), 8)
            for i, data in received:
                np.testing.assert_array_equal(data, actions[i, :[20, 50, 10, 50][i]])

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
