"""Dojo step timeout and PhysX device memory recovery."""
import tempfile
import unittest
from pathlib import Path
from pair_gate import oom_evidence
from oom_classify_1127 import driver_failure

ERROR = "[Error] [omni.physx.plugin] PhysX ABORT error: PhysX failed to allocate GPU memory - aborting simulation."
TIMEOUT = "TimeoutError: Dojo step timed out on GPU 5"

class DojoPhysX(unittest.TestCase):
    def check(self, driver, native, gpu=5):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "formal-driver.log").write_text(driver + "\n")
            (directory / f"native-g{gpu}.log").write_text(native + "\n")
            return oom_evidence(directory, {}, {"phase": "FAILED_RETURN_RLT"}, 0, "dojo", True)

    def test_matches_failing_card(self):
        result = self.check('"rpc_timeout": 900,\n' + TIMEOUT, ERROR)
        self.assertEqual((result["kind"], result["log"], result["driver_rpc_error"]),
                         ("physx_gpu_oom", "native-g5.log", TIMEOUT))

    def test_other_gpu(self):
        self.assertIsNone(self.check(TIMEOUT, ERROR, gpu=6))

    def test_timeout_without_allocation_error(self):
        self.assertIsNone(self.check(TIMEOUT, "PhysX has reported too many errors, simulation has been stopped."))

    def test_timeout_configuration_without_runtime_failure(self):
        self.assertIsNone(self.check('"rpc_timeout": 900,', ERROR))

    def test_runtime_error_only(self):
        self.assertIsNone(self.check("ValueError: invalid layout", ERROR))

    def test_completed_round_discards_old_error(self):
        self.assertIsNone(driver_failure([TIMEOUT, "│ Global Step:  12/300 │ Progress: 4%"])[2])

if __name__ == "__main__":
    unittest.main()

