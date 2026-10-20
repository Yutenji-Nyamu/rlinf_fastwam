"""A changed weak checkpoint must not reuse original-policy calibration keys."""
import json
import tempfile
import unittest
from pathlib import Path

from calibration_pool import collect, deduplicate
from threshold_sweep import conditions


class WeakOverlayTests(unittest.TestCase):
    def test_calibration_identity_contains_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, overlay in (("original", None), ("clean", "/protected/clean.pt")):
                batch = root / "batches" / label
                batch.mkdir(parents=True)
                config = {"num_steps": 10, "execute_chunk": 20,
                          "model": {"model_path": "/base", "openpi": {"action_chunk": 50}},
                          "env": {"task_config": {"task_name": "move_pillbottle_pad"}}}
                if overlay:
                    config["weak_weights"] = overlay
                decisions = [{"slot": 0, "query": 0, "actual_seed": 42, "noise_seed": 42,
                              "start": 0, "end": 20, "strong": False, "u": 1.0}]
                for name, value in (("config.json", config), ("decisions.json", decisions), ("done.json", {})):
                    (batch / name).write_text(json.dumps(value))
            pool = deduplicate([collect([root], 20)])
            self.assertEqual(pool["unique_decisions"], 2)

    def test_ten_thresholds_per_direction(self):
        rows = conditions([{"u": x / 1000} for x in range(1000)],
                          fractions=(.05, .15, .25, .35, .45, .55, .65, .75, .85, .95))
        self.assertEqual(len(rows), 20)
        for row in rows:
            self.assertAlmostEqual(row["calibration_fraction"], row["target_fraction"], places=6)


if __name__ == "__main__":
    unittest.main()
