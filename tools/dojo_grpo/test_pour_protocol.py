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

