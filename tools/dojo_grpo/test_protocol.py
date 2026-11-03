"""CPU checks for group identity, split separation, and terminal credit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rlinf/envs/robodojo"))
from protocol import (
    grouped_indices,
    layout_pools,
    terminal_arrays,
    replace_invalid_layouts,
)
from initial_state import audit_objects


class Contract(unittest.TestCase):
    def test_native_reset_failure_flags(self):
        from types import SimpleNamespace
        from native_entry import invalid_reset_slots

        env = SimpleNamespace(
            unstable_envs={0},
            success=[True, False, True, True],
            end_flag=[False, True, False, False],
            scene_manager=SimpleNamespace(
                layout_manager=SimpleNamespace(layout_valid=[True, True, False, True])
            ),
        )
        self.assertEqual(invalid_reset_slots(env, 4), {0, 1, 2})

    def test_physics_skip_preserves_groups(self):
        ids = [1] * 8 + [3] * 8
        result, changes = replace_invalid_layouts(ids, {1}, 55, 8)
        self.assertEqual(result[8:], [3] * 8)
        self.assertEqual(len(set(result[:8])), 1)
        self.assertNotIn(result[0], [1, 3])
        self.assertEqual(len(changes), 1)

    def test_object_identity_ignores_runtime_counter(self):
        from types import SimpleNamespace

        class Manager:
            instance_type_by_env = [{"bottle_3_18": "rigid"}, {"bottle_3_29": "rigid"}]
            gap = 0.0
            rotation_gap = 0.0

            def decode_inst_name(self, name):
                model, variant, _ = name.rsplit("_", 2)
                return model, int(variant)

            def get_instance_pose(self, env_idx, **kwargs):
                angle = self.rotation_gap if env_idx == 3 else 0.0
                return np.array([self.gap * env_idx, 0.0, 0.0]), np.array(
                    [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
                )

        manager = Manager()
        env = SimpleNamespace(scene_manager=SimpleNamespace(layout_manager=manager))
        self.assertEqual(audit_objects(env, 2, 2)["object_count"], 1)
        manager.gap = 0.02
        with self.assertRaises(RuntimeError):
            audit_objects(env, 2, 2)
        manager.gap = 0.0
        manager.rotation_gap = np.deg2rad(10)
        manager.instance_type_by_env += [
            {"bottle_3_30": "rigid"},
            {"bottle_3_31": "rigid"},
        ]
        drift = audit_objects(env, 4, 2, raise_on_mismatch=False)
        self.assertEqual(drift["bad_slots"], [2, 3])
        self.assertAlmostEqual(drift["rotation_max_deg"], 10)

    def test_groups(self):
        ids = grouped_indices(55, 32, 8, np.random.RandomState(42))
        self.assertEqual(len(ids), 32)
        self.assertTrue(all(len(set(ids[i : i + 8])) == 1 for i in range(0, 32, 8)))
        with self.assertRaises(ValueError):
            grouped_indices(55, 30, 8, np.random.RandomState(42))

    def test_terminal_once(self):
        d = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1]], bool)
        s = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], bool)
        r, t, x = terminal_arrays([False, False, True], d, s)
        np.testing.assert_array_equal(r, [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_array_equal(t | x, d)

    def test_layout_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in [0, 1, 2]:
                (root / str(seed)).mkdir()
                for i in [0, 1]:
                    (root / str(seed) / f"task_{i}.json").write_text(
                        json.dumps({"pose": seed + i})
                    )
            pools = layout_pools(root, "task")
            a = {x["digest"] for x in pools["train"]}
            b = {x["digest"] for x in pools["eval"]}
            self.assertFalse(a & b)
            self.assertTrue(a and b)


if __name__ == "__main__":
    unittest.main()
