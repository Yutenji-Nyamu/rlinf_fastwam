"""CPU checks for formal-RLT continuation and the two live owner wiring patterns."""
from pathlib import Path
from types import SimpleNamespace as Box
from unittest import TestCase, main, mock
import json
import sys
import tempfile

from patch_formal_owners import update

import rlt_formal_oom_1127 as policy


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
        inputs = Path(__file__).parent / "source"
        import os
        named = os.environ.get("OOM_RETRY_OWNER_SOURCES")
        if named:
            owners = [Path(p) for p in named.split(os.pathsep)]
        else:
            owners = sorted([*inputs.rglob("rt2_owner.py"), *inputs.rglob("rlt_dojo_owner.py")])
        self.assertTrue(owners)
        for owner in owners:
            mode = "rt2" if owner.name == "rt2_owner.py" else "dojo"
            patched = update(owner.read_text(encoding="utf-8-sig"), mode)
            compile(patched, str(owner), "exec")
            self.assertEqual(update(patched, mode), patched)
            self.assertIn("retry_formal_gpu_oom", patched)


if __name__ == "__main__":
    main()

