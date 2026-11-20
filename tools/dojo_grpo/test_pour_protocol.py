"""The official single-seed score uses one outcome per distinct held-out layout."""
import json
import tempfile
import unittest
from pathlib import Path

from pour_protocol import standard_evaluation


def wave(mode, end, ids, successes):
    return json.dumps({"mode": mode, "time": end, "rows": [
        {"layout": idx, "success": idx in successes, "actions": 400} for idx in ids
    ]}) + "\n"


class StandardEvaluationTest(unittest.TestCase):
    def test_first_25_current_eval_across_both_cards(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            left, right = [root / f"native-g{g}" for g in (5, 6)]
            left.mkdir()
            right.mkdir()
            (left / "completed-waves.jsonl").write_text(
                wave("eval", 1, range(16), set(range(16)))
                + wave("train", 3, range(16), set(range(16)))
                + wave("eval", 4, reversed(range(16)), {0}))
            (right / "completed-waves.jsonl").write_text(
                wave("eval", 4, range(16, 32), {16, 24, 25, 31})
                + wave("eval", 4, [0], set()))
            result = standard_evaluation(root, [5, 6], 2, 25)
            self.assertEqual(result["successes"], 3)
            self.assertEqual(result["distinct_completed"], 32)
            self.assertEqual([r["layout"] for r in result["rows"]], list(range(25)))
            with self.assertRaisesRegex(ValueError, "32/33"):
                standard_evaluation(root, [5, 6], 2, 33)

    def test_pouring_groups_keep_position_and_native_settling_limits(self):
        import math
        from types import SimpleNamespace

        from initial_state import audit_objects
        from native_entry import group_rotation_limit

        def audit(dx, angle):
            a = math.radians(angle) / 2
            poses = [([0, 0, 0], [1, 0, 0, 0]), ([dx, 0, 0], [math.cos(a), 0, 0, math.sin(a)])]
            manager = SimpleNamespace(instance_type_by_env=[{"b": "rigid"}] * 2,
                                      decode_inst_name=lambda _: ("bottle",),
                                      get_instance_pose=lambda env_idx, **kw: poses[env_idx])
            env = SimpleNamespace(scene_manager=SimpleNamespace(layout_manager=manager))
            return audit_objects(env, 2, 2, raise_on_mismatch=False,
                                 rotation_limit_deg=group_rotation_limit("pour_liquid_into_cup", "train"))
        self.assertEqual(group_rotation_limit("put_bottles_into_dustbin", "train"), 2.0)
        self.assertEqual(group_rotation_limit("pour_liquid_into_cup", "eval"), 2.0)
        self.assertEqual(audit(0.004, 5.33)["bad_slots"], [])
        self.assertEqual(audit(0.006, 5.33)["bad_slots"], [0, 1])
        self.assertEqual(audit(0.004, 11.0)["bad_slots"], [0, 1])

