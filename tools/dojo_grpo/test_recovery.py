# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""CPU checks for the only new outer recovery decisions; no GPU processes."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from recover_queue import latest_checkpoint, prepare_retry, simulation_failure


class RecoveryTests(unittest.TestCase):
    def test_simulation_failure_not_operator_stop_or_optimizer_error(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            (d / "status.json").write_text(json.dumps({"error": "child exited 1"}))
            log = d / "formal-driver.log"
            log.write_text("RuntimeError: CUDA out of memory")
            self.assertFalse(simulation_failure(d))
            log.write_text("TimeoutError: Dojo step timed out on GPU 7")
            self.assertTrue(simulation_failure(d))
            (d / "status.json").write_text(json.dumps({"error": "SystemExit: 143"}))
            self.assertFalse(simulation_failure(d))

    def test_complete_checkpoint_selection_and_retry_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "attempt"
            (old / "config").mkdir(parents=True)
            (old / "rpc").mkdir()
            (old / "rpc/authkey").write_bytes(b"test-only-key")
            plan = {"run": str(root / "results"), "model": "test-model"}
            for name in (
                "plan.json",
                "training-environment.json",
                "sim-environment.json",
                "launch-formal.json",
            ):
                (old / name).write_text(json.dumps(plan if name == "plan.json" else {}))
            for gpu in (6, 7):
                (old / f"native-g{gpu}/cache").mkdir(parents=True)
                (old / f"native-g{gpu}.json").write_text(
                    json.dumps(
                        {
                            "socket": str(old / f"rpc/gpu{gpu}.sock"),
                            "invalid_layouts": {"train": [32], "eval": [1]},
                        }
                    )
                )
            (old / "native-g7/invalid-resets.jsonl").write_text(
                json.dumps({"mode": "train", "invalid_layouts": [36]}) + "\n"
            )
            cfg = {
                "runner": {"resume_dir": None, "save_interval": 20, "max_steps": 200},
                "env": {"train": {"seed": 42}, "eval": {"seed": 42}},
                "algorithm": {
                    "dvac_gradient_weighting": {"output_dir": "old-diagnostics"}
                },
            }
            (old / "config/formal.yaml").write_text(yaml.safe_dump(cfg))
            checkpoints = Path(plan["run"]) / "clean-dustbin/checkpoints"
            for step in (2, 3, 4):
                cp = checkpoints / f"global_step_{step}"
                (cp / "actor/model_state_dict").mkdir(parents=True)
                if step != 4:  # 4 was interrupted before the complete marker.
                    (cp / "dojo-complete.json").write_text("{}")
                if step != 2:  # 2 was pruned after 3 completed.
                    (cp / "actor/model_state_dict/full_weights.pt").write_bytes(
                        b"test-weight"
                    )
            complete = checkpoints / "global_step_3"
            self.assertEqual(latest_checkpoint(plan), complete)
            new, new_plan = prepare_retry(old, old, 1)
            actual = yaml.safe_load((new / "config/formal.yaml").read_text())
            self.assertEqual(actual["runner"]["resume_dir"], str(complete))
            self.assertEqual(actual["runner"]["max_steps"], 200)
            self.assertEqual(actual["runner"]["save_interval"], 1)
            self.assertNotEqual(actual["env"]["train"]["seed"], 42)
            self.assertEqual(actual["env"]["eval"]["seed"], 42)
            self.assertEqual(new_plan["accepted_smoke"], str(old))
            native = json.loads((new / "native-g6.json").read_text())
            self.assertEqual(native["invalid_layouts"]["train"], [32, 36])
            self.assertEqual(native["socket"], str(new / "rpc/gpu6.sock"))
            self.assertTrue(complete.is_dir())


if __name__ == "__main__":
    unittest.main()
