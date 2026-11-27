"""CPU checks for formal-RLT continuation and the two live owner wiring patterns."""
from pathlib import Path
from types import SimpleNamespace as Box
from unittest import TestCase, main, mock
import ast
import json
import os
import sys
import tempfile

from patch_formal_owners import intermediate_owner, update

import rlt_formal_oom_1127 as policy


def source_owners():
    named = os.environ.get("OOM_RETRY_OWNER_SOURCES")
    if named:
        return [Path(p) for p in named.split(os.pathsep)]
    inputs = Path(__file__).parent / "source"
    return sorted([*inputs.rglob("rt2_owner.py"), *inputs.rglob("rlt_dojo_owner.py")])


class FormalRetryTests(TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.base = Path(td.name)
        self.root = self.base / "run/checkpoints"
        self.cfg = Box(runner=Box(logger=Box(log_path=str(self.base), experiment_name="run"), resume_dir=None),
                       actor=Box(fsdp_config=Box(use_orig_params=False)),
                       algorithm=Box(entropy_tuning=Box(alpha_type="fixed_alpha")))

    def cp(self, step, *, complete=True, local=False):
        path = self.root / f"global_step_{step}"
        files = {"dojo-complete.json": json.dumps({"step": step}),
                 "actor/sac_components/rlt_trainer_state/complete.json":
                     json.dumps(dict(complete=complete, saved_runner_step=step, actor_world_size=1)),
                 "actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt": "state",
                 "actor/sac_components/target_model/checkpoint_rank_0.pt": "target",
                 "actor/sac_components/replay_buffer/rank_0/metadata.json":
                     json.dumps({"size": 2, "trajectory_format": "pt"}),
                 "actor/sac_components/replay_buffer/rank_0/trajectory_index.json": json.dumps({
                     "trajectory_id_list": [0, 1], "trajectory_index": {
                         "0": {"model_weights_id": "old"}, "1": {"model_weights_id": "latest"}}}),
                 "actor/sac_components/replay_buffer/rank_0/trajectory_1_latest.pt": "sample"}
        if local:
            files["actor/local_shard_checkpoint/checkpoint_rank_0.pt"] = "model and optimizer"
        else:
            files.update({"actor/dcp_checkpoint/.metadata":"index","actor/dcp_checkpoint/__0_0.distcp":"data"})
        for name, data in files.items():
            dest = path / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(data)
        return path

    def test_partial_later_step_and_cleaned_replay_are_skipped(self):
        old = self.cp(116)
        self.cp(118, complete=False)
        newer = self.cp(120)
        (newer / "actor/sac_components/replay_buffer/rank_0/metadata.json").unlink()
        self.assertEqual(policy.select_retry(self.cfg), (True, old))

    def test_initial_fresh_and_missing_explicit_source(self):
        self.assertEqual(policy.select_retry(self.cfg), (True, None))
        self.cfg.runner.resume_dir = str(self.root / "global_step_8")
        self.assertEqual(policy.select_retry(self.cfg), (False, None))

    def test_replay_cleanup_with_index_and_an_older_sample_still_present(self):
        previous = self.cp(4)
        newer = self.cp(8)
        replay = newer / "actor/sac_components/replay_buffer/rank_0"
        latest = replay / "trajectory_1_latest.pt"
        (replay / "trajectory_0_old.pt").write_bytes(b"older sample")
        latest.unlink()
        self.assertEqual(policy.select_retry(self.cfg), (True, previous))
        latest.touch()
        self.assertEqual(policy.select_retry(self.cfg), (True, previous))

    def test_external_prior_with_old_sample_rolled_out(self):
        source = self.cp(4)
        archived = self.base / "other-run/checkpoints/global_step_4"
        archived.parent.mkdir(parents=True)
        source.rename(archived)
        self.cfg.runner.resume_dir = str(archived)
        self.assertEqual(policy.select_retry(self.cfg), (True, archived))

    def test_local_shard_must_match_loader(self):
        older = self.cp(4)
        local = self.cp(8, local=True)
        self.assertEqual(policy.select_retry(self.cfg), (True, older))
        self.cfg.actor.fsdp_config.use_orig_params = True
        self.assertEqual(policy.select_retry(self.cfg), (True, local))

    def test_native_log_offset_starts_at_this_attempt(self):
        path = self.base / "native.log"
        path.write_bytes(b"out of memory\n")
        old = path.stat().st_size
        with path.open("ab") as log:
            log.write(b"new error\n")
        self.assertEqual(list(policy.read_attempt(path, old)), ["new error\n"])

    def test_explicit_oom_changes_only_resume_dir_and_other_exits_do_not(self):
        checkpoint = self.cp(18)
        (self.base / "config").mkdir()
        config = self.base / "config/formal.yaml"
        config.write_text("old config")
        cfg = self.cfg
        writer = Box(load=lambda _: cfg, save=lambda value, path: path.write_text(str(value.runner.resume_dir)))
        save = mock.Mock()
        with mock.patch.dict(sys.modules, omegaconf=Box(OmegaConf=writer), common=Box(save=save)):
            output = self.base / "formal.log"
            output.write_text("ConnectionRefusedError: socket has no listener\n")
            self.assertIsNone(policy.prepare_oom_retry(self.base, output))
            self.assertEqual(config.read_text(), "old config")
            output.write_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80 MiB\n")
            self.assertTrue(policy.prepare_oom_retry(self.base, output))
            self.assertEqual(config.read_text(), str(checkpoint))
            self.assertEqual(save.call_args.args[1]["kind"], "cuda")

    def test_old_driver_oom_is_cleared_by_real_progress_table(self):
        output = self.base / "formal.log"
        output.write_text("RuntimeError: CUDA out of memory.\n"
                          "│ Global Step:   89/300 │ Progress: 29.7%\n"
                          "KeyError: invalid task\n", encoding="utf8")
        with mock.patch.dict(sys.modules, omegaconf=Box(OmegaConf=mock.Mock()), common=Box(save=mock.Mock())):
            self.assertIsNone(policy.prepare_oom_retry(self.base, output))

    def test_native_oom_needs_current_driver_disconnect_and_ordered_time(self):
        output = self.base / "formal.log"
        native = self.base / "native-g6.log"
        native.write_text("2026-11-26T22:00:00Z RuntimeError: CUDA out of memory.\n")
        output.write_text("KeyError: invalid task\n")
        config_dir = self.base / "config"
        config_dir.mkdir()
        cfg = self.cfg
        writer = Box(load=lambda _: cfg, save=lambda value, path: path.write_text(str(value.runner.resume_dir)))
        with mock.patch.dict(sys.modules, omegaconf=Box(OmegaConf=writer), common=Box(save=mock.Mock())):
            self.assertIsNone(policy.prepare_oom_retry(self.base, output, (native, 0)))
            output.write_text("│ Global Step:   89/300 │ Progress: 29.7%\nConnectionResetError: peer closed\n", encoding="utf8")
            cp = self.cp(89)
            (cp / "dojo-complete.json").write_text('{"step":89,"time":1795734000}')
            self.assertIsNone(policy.prepare_oom_retry(self.base, output, (native, 0)))
            native.write_text("2026-11-28T22:00:00Z RuntimeError: CUDA out of memory.\n")
            self.assertTrue(policy.prepare_oom_retry(self.base, output, (native, 0)))

    def test_real_owners_can_be_patched_twice(self):
        owners = source_owners()
        self.assertTrue(owners)
        for owner in owners:
            mode = "rt2" if owner.name == "rt2_owner.py" else "dojo"
            patched = update(owner.read_text(encoding="utf-8-sig"), mode)
            compile(patched, str(owner), "exec")
            self.assertEqual(update(patched, mode), patched)
            before = owner.read_text(encoding="utf-8-sig")
            if "gpu_oom" not in before:
                self.assertEqual(update(intermediate_owner(before, mode), mode), patched)
            self.assertIn("retry_formal_gpu_oom", patched)

    def test_repeated_external_oom_and_unknown_oom_at_same_or_new_step(self):
        import collections
        cp = self.cp(18)
        (self.base / "config").mkdir()
        log = self.base / "formal.log"
        log.write_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80 MiB\n")
        external = collections.deque([True, True, False, False, False])
        def decide(evidence, cards, count, streak):
            self.assertEqual(cards, [7])
            self.assertEqual(evidence["physical_gpu"], 7)
            contested = external.popleft()
            return (1 if contested or not count else None), {"external": contested}
        writer = Box(load=lambda _: self.cfg, save=lambda value, path: path.write_text(str(value.runner.resume_dir)))
        budget = {}
        with mock.patch.dict(sys.modules, omegaconf=Box(OmegaConf=writer), common=Box(save=mock.Mock()),
                             oom_contention=Box(retry_delay=mock.Mock(side_effect=decide))):
            self.assertTrue(policy.prepare_oom_retry(self.base, log, budget=budget, gpu=7))
            self.assertTrue(policy.prepare_oom_retry(self.base, log, budget=budget, gpu=7))
            self.assertFalse(policy.prepare_oom_retry(self.base, log, budget=budget, gpu=7))
            self.assertEqual(budget["key"], ("formal", str(cp)))
            newer = self.cp(20)
            self.assertTrue(policy.prepare_oom_retry(self.base, log, budget=budget, gpu=7))
            self.assertFalse(policy.prepare_oom_retry(self.base, log, budget=budget, gpu=7))
            self.assertEqual(budget["key"], ("formal", str(newer)))

    def test_pretraining_retry_reuses_its_own_entry_and_separate_logs(self):
        path = self.base / "stage1.log"
        path.write_text("torch.OutOfMemoryError: CUDA out of memory.\n")
        writer = Box(load=lambda _: self.cfg, save=mock.Mock())
        with mock.patch.dict(sys.modules, omegaconf=Box(OmegaConf=writer), common=Box(save=mock.Mock()),
                             oom_contention=Box(retry_delay=lambda *args: (10, {"external": True}))):
            self.assertTrue(policy.prepare_oom_retry(self.base, path, budget={}, gpu=5, phase="stage1"))
        writer.save.assert_not_called()
        self.assertEqual([policy.attempt_name("stage1", n) for n in range(3)],
                         ["stage1", "stage1-retry", "stage1-retry-2"])

    def test_installed_rt2_owner_reopens_same_formal_after_two_contested_exits(self):
        owner = next(p for p in source_owners() if p.name == "rt2_owner.py")
        tree = ast.parse(update(owner.read_text(encoding="utf-8-sig"), "rt2"))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and ast.unparse(n.iter) == "p['phases']")
        compiled = compile(ast.Module(body=[loop], type_ignores=[]), str(owner), "exec")
        (self.base / "config").mkdir()
        self.cp(18)
        writer = Box(load=lambda _: self.cfg, save=lambda value, path: path.write_text(str(value.runner.resume_dir)))
        calls = []
        def start(argv, **kwargs):
            calls.append((argv, Path(kwargs["stdout"].name).name))
            failed = len(calls) < 3
            kwargs["stdout"].write("torch.OutOfMemoryError: CUDA out of memory.\n" if failed else "complete\n")
            return Box(wait=lambda **_: 1 if failed else 0, pid=len(calls))
        timer = Box(time=lambda: 1, sleep=mock.Mock())
        env = dict(p={"gpu": 4, "retry_formal_gpu_oom": True, "training_source": str(self.base),
                      "phases": [{"name": "formal", "command": ["formal-train"]}]},
                   d=self.base, sys=Box(path=[]), env={}, save=mock.Mock(), proc=lambda pid: pid,
                   time=timer, subprocess=Box(Popen=start, DEVNULL=None, STDOUT=None))
        with mock.patch.dict(sys.modules, rlt_retry=Box(prepare_retry=mock.Mock(side_effect=AssertionError)),
                             omegaconf=Box(OmegaConf=writer), common=Box(save=mock.Mock()),
                             oom_contention=Box(retry_delay=lambda *args: (5, {"external": True}))), \
             mock.patch.object(policy, "drain_formal") as drain:
            exec(compiled, env)
        self.assertEqual(calls, [(["formal-train"], "formal.log"), (["formal-train"], "formal-retry.log"),
                                 (["formal-train"], "formal-retry-2.log")])
        self.assertEqual(timer.sleep.call_args_list, [mock.call(5), mock.call(5)])
        self.assertEqual(drain.call_count, 2)


if __name__ == "__main__":
    main()

