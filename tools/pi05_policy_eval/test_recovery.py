"""Regression checks for finite recovery and preserving completed results."""

import tempfile
import unittest
from pathlib import Path

from recovery import archive_failed_batch, limits


class RecoveryTests(unittest.TestCase):
    def test_resume_scientific_numbers_keep_numeric_types(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.create("log_eps: 1e-12\nminmax_eps: 1e-06\n"))
        self.assertEqual(cfg, dict(log_eps=1e-12, minmax_eps=1e-6))

    def test_default_keeps_existing_single_attempt(self):
        self.assertEqual(limits({})["attempts"], 1)
        self.assertFalse(limits({})["continue_errors"])

    def test_pair_wait_covers_both_batches_and_retries(self):
        plan = {"recovery": dict(attempts=2, batch_seconds=10800,
                                stall_seconds=1800, barrier_seconds=43800,
                                continue_errors=True)}
        self.assertEqual(limits(plan)["attempts"], 2)
        plan["recovery"]["barrier_seconds"] = 7500
        with self.assertRaises(AssertionError):
            limits(plan)

    def test_archive_retains_failure_but_never_completed_or_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batches" / "condition_b0"
            batch.mkdir(parents=True)
            (batch / "error.json").write_text("failure")
            archived = archive_failed_batch(root, batch, 0)
            self.assertEqual((Path(archived) / "error.json").read_text(), "failure")
            self.assertFalse(batch.exists())
            batch.mkdir()
            (batch / "done.json").write_text("{}")
            with self.assertRaises(AssertionError):
                archive_failed_batch(root, batch, 1)
            with self.assertRaises(AssertionError):
                archive_failed_batch(root, root / "other", 0)


if __name__ == "__main__":
    unittest.main()
