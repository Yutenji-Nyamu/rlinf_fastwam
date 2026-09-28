"""CPU-only checks for cutover identity and sequencing; no server access."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import resource  # noqa: F401
except ImportError:
    sys.modules["resource"] = types.ModuleType("resource")
spec = importlib.util.spec_from_file_location("six_task_ops", Path(__file__).with_name("ops.py"))
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class CutoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        ops.ST = Path(self.tmp.name)
        self.identity = {"pid": 100, "uid": 20001, "start": 300, "state": "S", "ppid": 1}
        self.old = {"namespace": "old_ns", "run": "/run/old", "identity": self.identity, "gpus": [4, 5]}
        self.plan = {"uid": 20001, "old_runs": [self.old]}

    def test_pid_reuse_is_not_same_process(self):
        with patch.object(ops, "proc", return_value={**self.identity, "start": 301}):
            self.assertFalse(ops.same(self.identity))
        with patch.object(ops, "proc", return_value=self.identity):
            self.assertTrue(ops.same({"pid": 100, "uid": 20001, "start_ticks": 300}))

    def test_foreign_gpu_process_blocks_before_signal_or_receipt(self):
        with patch.object(ops, "checked", return_value=self.plan), \
             patch.object(ops, "active_actors", return_value=[]), \
             patch.object(ops, "proc", return_value=self.identity), \
             patch.object(ops, "process_tree", return_value={100: self.identity}), \
             patch.object(ops, "gpu_pids", return_value=[999]), \
             patch.object(ops.os, "kill") as kill:
            with self.assertRaisesRegex(AssertionError, "Unrelated GPU"):
                ops.stop_old()
            kill.assert_not_called()
        self.assertFalse((ops.ST / "old-stop-attempt.json").exists())

    def test_actor_job_mismatch_blocks_cleanup(self):
        actor = {"name": "worker", "pid": 100, "job_id": "foreign", "actor_id": "a"}
        with self.assertRaisesRegex(AssertionError, "Unexpected job"):
            ops.validate_scoped_actors(self.plan, [actor], {"expected"})

    def test_stage1_failure_keeps_old_training(self):
        child = types.SimpleNamespace(wait=lambda: 1)
        self.plan["runs"] = {"stage1-full": {"run": str(ops.ST / "s1")}}
        (ops.ST / "s1/runtime").mkdir(parents=True)
        (ops.ST / "plan.json").write_text(json.dumps(self.plan))
        with patch.object(ops, "checked", return_value=self.plan), \
             patch.object(ops, "proc", return_value=self.identity), \
             patch.object(ops, "launch", return_value=child), \
             patch.object(ops, "stop_old") as stop:
            with self.assertRaises(AssertionError):
                ops.pipeline()
            stop.assert_not_called()

    def test_pipeline_stage1_then_stop_then_formals(self):
        order = []
        child = types.SimpleNamespace(pid=100, poll=lambda: 0)
        def launch(key):
            order.append("launch:" + key)
            return child
        with patch.object(ops, "checked", return_value=self.plan), \
             patch.object(ops, "proc", return_value=self.identity), \
             patch.object(ops, "launch", side_effect=launch), \
             patch.object(ops, "wait_stage1", side_effect=lambda _: order.append("stage1:complete")), \
             patch.object(ops, "stop_old", side_effect=lambda: order.append("old:stop")), \
             patch.object(ops, "finished"):
            ops.pipeline()
        self.assertEqual(order, ["launch:stage1-full", "stage1:complete", "old:stop", "launch:clean", "launch:combo"])
        self.assertTrue((ops.ST / "formal-dispatched.json").exists())

    def test_duplicate_receipt_is_never_overwritten(self):
        target = ops.ST / "old-stop-attempt.json"
        ops.save(target, {"first": True})
        with self.assertRaises(FileExistsError):
            ops.save(target, {"second": True})
        self.assertEqual(json.loads(target.read_text()), {"first": True})

    def test_graceful_stop_needs_no_external_ray_or_kill_escalation(self):
        alive = [True]
        sent = []
        def same(_):
            return alive[0]
        def kill(pid, signum):
            sent.append((pid, signum))
            alive[0] = False
        with patch.object(ops, "checked", return_value=self.plan), \
             patch.object(ops, "active_actors", return_value=[]), \
             patch.object(ops, "same", side_effect=same), \
             patch.object(ops, "process_tree", side_effect=lambda roots, _: {100: self.identity} if roots else {}), \
             patch.object(ops, "gpu_pids", side_effect=lambda _: [100] if alive[0] else []), \
             patch.object(ops.os, "kill", side_effect=kill), \
             patch.object(ops.signal, "SIGKILL", 9, create=True), \
             patch.object(ops, "kill_named_actors") as actor_kill:
            ops.stop_old()
            actor_kill.assert_not_called()
        self.assertEqual(sent, [(100, signal.SIGTERM)])
        self.assertTrue((ops.ST / "old-stopped.json").exists())


if __name__ == "__main__":
    unittest.main()
