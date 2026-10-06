"""CPU-only lease regression checks; run in the SZ1 project Python, never locally.

No SSH, Ray, model, /proc, nvidia-smi, or real deployment paths are accessed.
"""
import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import yaml

spec = importlib.util.spec_from_file_location('lease_under_test', Path(__file__).with_name('rlt_lease.py'))
lease = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lease)


def save(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.common = SimpleNamespace(save=save)

    def checkpoint(self, run, step, *, complete=True, samples=3):
        cp = run / run.name / 'checkpoints' / f'global_step_{step}'
        state = cp / 'actor/sac_components/rlt_trainer_state'
        state.mkdir(parents=True)
        save(state / 'complete.json', dict(complete=complete, saved_runner_step=step,
                                          actor_world_size=1, update_step=42))
        if not complete:
            return cp
        replay = cp / 'actor/sac_components/replay_buffer/rank_0'
        replay.mkdir(parents=True)
        save(replay / 'metadata.json', dict(total_samples=samples))
        save(replay / 'trajectory_index.json', dict(trajectory_id_list=list(range(samples))))
        dcp = cp / 'actor/dcp_checkpoint'
        dcp.mkdir()
        (dcp / '.metadata').write_bytes(b'cpu-fixture-metadata')
        target = cp / 'actor/sac_components/target_model'
        target.mkdir()
        (target / 'checkpoint_rank_0.pt').write_bytes(b'cpu-fixture-target')
        return cp

    def return_fixture(self):
        old, new, stage = self.root / 'original-run', self.root / 'return-run', self.root / 'lease'
        (old / 'runtime').mkdir(parents=True)
        stage.mkdir()
        self.checkpoint(old, 75)
        checkpoint = lease._checkpoint(old)
        save(stage / 'rlt-released.json', dict(checkpoint=checkpoint))
        cfg = dict(
            runner=dict(max_epochs=3000, max_steps=3000, resume_dir=None,
                        save_interval=25, val_check_interval=25,
                        logger=dict(log_path=str(old), experiment_name=old.name)),
            env=dict(train=dict(total_num_envs=8, max_episode_steps=200,
                                task_config=dict(task_name='place_fan', save_path=str(old / 'data'))),
                     eval=dict(total_num_envs=4)),
            actor=dict(global_batch_size=512, micro_batch_size=256, seed=1234),
            algorithm=dict(update_epoch=5, gamma=0.99, rlt_schedule=dict(warmup_min_size=20000)),
        )
        (old / 'runtime/resolved.yaml').write_text(yaml.safe_dump(cfg))
        env = dict(RLT_LOG_ROOT=str(old), CLUSTER_NAMESPACE='old-gpu4',
                   HOME='/home/chenyiteng', RLINF_OPENDW_GPU_SCOPE_MANIFEST='/fixed/scope-gpu4.json')
        save(old / 'runtime/environment.json', env)
        row = dict(run=str(old), namespace='old-gpu4', gpus=[4])
        untouched = dict(run='/untouched/gpu5', namespace='keep-gpu5', gpus=[5])
        plan = dict(runs={'clean': row, 'combo': untouched, 'stage1-full': {'run': '/stage1', 'gpus': [2]}})
        bound = dict(row=row, new_run=str(new), new_namespace='new-gpu4', role='clean')
        return stage, old, new, cfg, plan, bound

    def test_only_authorized_cards_can_prepare(self):
        for gpu in (0, 3, 6, 7):
            stage = self.root / f'forbidden-{gpu}'
            with self.subTest(gpu=gpu), self.assertRaises(AssertionError):
                lease.prepare(stage, gpu=gpu, operation_id='test',
                              new_run='/unused', new_namespace='unused')
            self.assertFalse(stage.exists(), 'Reject scope before creating lease files')

    def test_latest_complete_checkpoint_and_frozen_identity(self):
        run = self.root / 'checkpoint-run'
        complete = self.checkpoint(run, 75)
        self.checkpoint(run, 100, complete=False)
        result = lease._checkpoint(run)
        self.assertEqual((result['step'], result['path'], result['replay_samples']), (75, str(complete), 3))
        lease._check_checkpoint(result)
        Path(result['target']['path']).write_bytes(b'modified-target-with-different-size')
        with self.assertRaises(AssertionError):
            lease._check_checkpoint(result)

    def test_corrupt_completed_checkpoint_does_not_silently_fall_back(self):
        run = self.root / 'corrupt-run'
        self.checkpoint(run, 50)
        newest = self.checkpoint(run, 75)
        save(newest / 'actor/sac_components/replay_buffer/rank_0/metadata.json', dict(total_samples=99))
        with self.assertRaises(AssertionError):
            lease._checkpoint(run)

    def test_return_keeps_budget_method_scope_and_other_role(self):
        stage, old, new, cfg, plan, bound = self.return_fixture()
        before_bytes = (old / 'runtime/resolved.yaml').read_bytes()
        plan_before = copy.deepcopy(plan)
        ret, returned_plan = lease._prepare_return(stage, bound, self.common, plan)
        actual = yaml.safe_load((new / 'runtime/resolved.yaml').read_text())
        self.assertEqual(actual['algorithm'], cfg['algorithm'])
        self.assertEqual(actual['actor'], cfg['actor'])
        self.assertEqual(actual['runner']['max_steps'], 3000)
        self.assertEqual(actual['env']['train']['total_num_envs'], 8)
        self.assertEqual(actual['env']['train']['max_episode_steps'], 200)
        self.assertEqual(actual['runner']['resume_dir'], lease._checkpoint(old)['path'])
        self.assertEqual(returned_plan['runs']['combo'], plan_before['runs']['combo'])
        self.assertEqual(returned_plan['runs']['stage1-full'], plan_before['runs']['stage1-full'])
        self.assertEqual(plan, plan_before)
        self.assertEqual((old / 'runtime/resolved.yaml').read_bytes(), before_bytes)
        environment = lease._read(new / 'runtime/environment.json')
        self.assertEqual(environment['RLINF_OPENDW_GPU_SCOPE_MANIFEST'], '/fixed/scope-gpu4.json')
        self.assertEqual(environment['CLUSTER_NAMESPACE'], 'new-gpu4')
        self.assertTrue(set(lease._read(ret / 'prepared.json')['config_changes']) <= lease.OUTPUT_KEYS)

    def test_return_rejects_changes_outside_output_whitelist(self):
        stage, old, new, cfg, plan, bound = self.return_fixture()
        cfg['algorithm']['unexpected_data_path'] = str(old / 'training-target')
        (old / 'runtime/resolved.yaml').write_text(yaml.safe_dump(cfg))
        with self.assertRaises(AssertionError):
            lease._prepare_return(stage, bound, self.common, plan)
        self.assertFalse(new.exists(), 'Reject unexpected config changes before creating the return run')

    def test_watch_preserves_concurrent_other_card_change(self):
        stage = self.root / 'watch-lease'
        stage.mkdir()
        watch = self.root / 'watch.json'
        original = dict(run='old4', namespace='old4-ns', gpus=[4])
        # GPU6 has already been legitimately reassigned by another window.
        other = dict(run='new-dsrl6', namespace='new-dsrl6-ns', gpus=[6])
        before = dict(runs={'target': original, 'other': other}, metadata={'keep': True})
        save(watch, before)
        bound = dict(watch=str(watch), watch_key='target', watch_entry=copy.deepcopy(original),
                     new_run='returned4', new_namespace='returned4-ns')
        lease._update_watch(stage, bound, self.common)
        actual = lease._read(watch)
        self.assertEqual(actual['runs']['other'], other)
        self.assertEqual(actual['metadata'], before['metadata'])
        self.assertEqual(actual['runs']['target'], dict(run='returned4', namespace='returned4-ns', gpus=[4]))
        lease._update_watch(stage, bound, self.common)  # Repeated inspection is idempotent.

    def test_watch_rejects_reassignment_of_target_card(self):
        stage = self.root / 'changed-target-lease'
        stage.mkdir()
        watch = self.root / 'changed-target.json'
        current = dict(runs={'target': dict(run='new-owner4', namespace='new-owner4-ns', gpus=[4])})
        save(watch, current)
        bound = dict(watch=str(watch), watch_key='target',
                     watch_entry=dict(run='old4', namespace='old4-ns', gpus=[4]),
                     new_run='returned4', new_namespace='returned4-ns')
        with self.assertRaises(AssertionError):
            lease._update_watch(stage, bound, self.common)
        self.assertEqual(lease._read(watch), current)
        self.assertFalse((stage / 'return-watch-before.json').exists())


if __name__ == '__main__':
    unittest.main()
