"""CPU-only classification of recording failures caught by the outer wrapper."""

import tempfile
import unittest
from pathlib import Path

from run_task import policy_task_error


class PolicyErrorTests(unittest.TestCase):
    def test_repeated_query_after_normal_output_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "policy.log"
            log.write_text("normal inference\n" * 2000
                           + "AssertionError: Repeated query identity\n")
            self.assertEqual(policy_task_error(log), "Repeated query identity")

    def test_missing_or_healthy_log_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "policy.log"
            self.assertIsNone(policy_task_error(log))
            log.write_text("Policy inference finished\n")
            self.assertIsNone(policy_task_error(log))


if __name__ == "__main__":
    unittest.main()
