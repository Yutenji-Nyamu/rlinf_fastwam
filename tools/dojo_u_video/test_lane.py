"""CPU-only failure routing tests; no simulator or GPU context."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lane
from common import read, save
from lane import accept_task


class FailureRoutingTests(unittest.TestCase):
    def test_cleared_device_failure_advances_only_when_enabled(self):
        self.assertTrue(accept_task(79, {"episodes": 0}, {"clear": True}, True))
        with self.assertRaises(RuntimeError):
            accept_task(79, {"episodes": 0}, {"clear": True}, False)

    def test_uncleared_device_never_advances(self):
        for clearance in (None, {"clear": False}):
            with self.subTest(clearance=clearance), self.assertRaises(RuntimeError):
                accept_task(79, {"episodes": 0}, clearance, True)

    def test_success_and_ordinary_skip_keep_existing_behavior(self):
        for state in ("COMPLETE", "PARTIAL", "SKIPPED"):
            self.assertFalse(accept_task(0, {"state": state}, {"clear": True}, True))

    def test_invalid_recording_still_stops(self):
        with self.assertRaises(RuntimeError):
            accept_task(78, {"state": "INVALID_RECORDING"}, {"clear": True}, True)

    def test_missing_result_still_stops(self):
        with self.assertRaises(RuntimeError):
            accept_task(79, None, {"clear": True}, True)


class LaneIntegrationTests(unittest.TestCase):
    def run_queue(self, codes, episode_counts=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for index in range(len(codes)):
                config = root / f"config-{index}.json"
                save(config, {"gpu": 7, "output": str(root / "tasks" / str(index))})
                jobs.append(str(config))
            save(root / "prior-release.json", {})
            save(root / "smoke-accepted.json", {"passed": True})
            save(
                root / "lane-g7-plan.json",
                {
                    "jobs": jobs,
                    "output": str(root),
                    "skip_cleared_device_fault": True,
                    "predecessor": {
                        "external_identity": {},
                        "external_release": str(root / "prior-release.json"),
                    },
                },
            )
            launched = []

            class FakeOps:
                def proc(self, pid):
                    return {"pid": pid}

                def same(self, identity):
                    return False

                def gpus(self):
                    return {7: {"processes": []}}

            class FakeChild:
                pid = 123

                def __init__(self, argv, **kwargs):
                    self.config = argv[-1]
                    self.code = codes[len(launched)]
                    self.episodes = (
                        episode_counts[len(launched)] if episode_counts is not None
                        else 25 if self.code == 0 else 0
                    )
                    launched.append(self.config)

                def wait(self):
                    out = Path(read(self.config)["output"])
                    save(out / "clearance.json", {"clear": True})
                    save(
                        out / "result.json",
                        {"episodes": self.episodes, "successes": 0},
                    )
                    return self.code

            with (
                patch.object(lane.sys, "argv", ["lane.py", str(root), "7"]),
                patch.object(lane.os, "getuid", return_value=20001),
                patch.object(lane.signal, "signal"),
                patch.object(lane, "ops", return_value=FakeOps()),
                patch.object(lane.subprocess, "Popen", FakeChild),
            ):
                lane.main()
            return (
                len(launched),
                read(root / "lane-g7-status.json"),
                (root / "lane-g7-released.json").exists(),
            )

    def test_device_fault_then_success_advances_and_releases(self):
        count, state, release = self.run_queue([79, 0])
        self.assertEqual(count, 2)
        self.assertEqual(state["phase"], "COMPLETE")
        self.assertTrue(release)

    def test_cleaned_policy_skip_advances_to_next_task(self):
        count, state, release = self.run_queue([0, 0], [0, 25])
        self.assertEqual(count, 2)
        self.assertEqual(state["phase"], "COMPLETE")
        self.assertTrue(release)

    def test_all_device_faults_finish_finite_queue_without_releasing_rlt(self):
        count, state, release = self.run_queue([79, 79])
        self.assertEqual(count, 2)
        self.assertEqual(state["phase"], "BLOCKED_DEVICE")
        self.assertFalse(release)


if __name__ == "__main__":
    unittest.main()
