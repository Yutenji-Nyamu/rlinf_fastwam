"""Focused server CPU fixtures; never launch an owner, GPU, Ray, or native env."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

import build_formal_config as B
import opendw_formal_owner as O


def configs():
    native = dict(env_type='robotwin', total_num_envs=32, rollout_epoch=1, group_size=1,
                  is_eval=True, use_fixed_reset_state_ids=True, auto_reset=True, ignore_terminations=True,
                  center_crop=False, task_config=dict(task_name='move_can_pot', embodiment=['aloha-agilex'],
                  camera=dict(collect_head_camera=True, collect_wrist_camera=True),
                  domain_randomization=dict(random_background=False)), video_cfg=dict(save_video=True))
    control = dict(runner=dict(max_epochs=1000, max_steps=200, save_interval=10), env=dict(eval=native))
    smoke = dict(cluster=dict(component_placement=B.PLACEMENT.copy()),
                 env=dict(group_name='old', train=dict(env_type='opendw_robotwin', task_name='adjust_bottle',
                     total_num_envs=64, rollout_epoch=8, group_size=8, chunk=32, max_episode_steps=384,
                     max_steps_per_rollout_epoch=384, service_urls=['http://127.0.0.1:1234', 'http://127.0.0.1:1235'],
                     video_cfg=dict(save_video=False, video_base_dir='/data/chenyiteng/old/video'))),
                 actor=dict(group_name='old', global_batch_size=2048, micro_batch_size=8),
                 algorithm=dict(update_epoch=2, dvac_gradient_weighting=dict(mode='off', output_dir='old')),
                 rollout=dict(group_name='old', model=dict(kind='same')),
                 weight_syncer={}, reward={}, critic={},
                 runner=dict(max_epochs=1, max_steps=1, resume_dir=None, ckpt_path=None,
                             logger=dict(log_path='/data/chenyiteng/old', experiment_name='old')))
    return smoke, control


def build(smoke, control):
    return B.build(smoke, control, name='formal-v1', run_dir='/data/chenyiteng/formal-v1/run',
                   services=['http://127.0.0.1:8126', 'http://127.0.0.1:8127'],
                   native_assets='/data/chenyiteng/robotwin', native_eval_seeds='/data/chenyiteng/seeds.json')


def startup_fixture():
    formal, _ = build(*configs())
    startup = copy.deepcopy(formal)
    startup['runner'].update(max_epochs=1, max_steps=1, save_interval=1, val_check_interval=1,
                             per_worker_log_path='/data/chenyiteng/formal-v1/startup_smoke/workers')
    startup['runner']['logger'].update(log_path='/data/chenyiteng/formal-v1/startup_smoke/run',
                                       experiment_name='startup-smoke')
    startup['env']['train'].update(rollout_epoch=1, max_episode_steps=32, max_steps_per_rollout_epoch=32)
    startup['env']['eval'].update(max_episode_steps=32, max_steps_per_rollout_epoch=32)
    startup['env']['eval']['task_config']['step_lim'] = 32
    startup['actor']['global_batch_size'] = 64
    row = dict(key='startup_smoke', episode_steps=32, num_envs=64, timeout_seconds=3600,
               namespace='opendw_startup', config='/data/chenyiteng/startup.yaml')
    return formal, startup, row


class FormalTests(unittest.TestCase):
    def test_startup_smoke_shadow_only_and_unchanged_actual_parameters(self):
        original = O.M
        try:
            formal, startup, row = startup_fixture()
            before = copy.deepcopy(startup)
            calls = []
            O.M = types.SimpleNamespace(validate_config=lambda *args: calls.append(args))
            O.validate_startup_smoke(startup, row, {}, Path('/data/chenyiteng/formal-v1'), formal)
            self.assertEqual(startup, before)
            self.assertEqual(len(calls), 1)
            shadow = calls[0][0]
            self.assertEqual(shadow['env']['train']['rollout_epoch'], 8)
            self.assertEqual(shadow['actor']['global_batch_size'], 512)
            self.assertEqual(shadow['runner']['val_check_interval'], -1)
            self.assertEqual(startup['env']['train']['rollout_epoch'], 1)
            self.assertEqual(startup['actor']['global_batch_size'], 64)
            self.assertEqual(startup['env']['eval']['total_num_envs'], 32)
            self.assertEqual(startup['env']['eval']['max_episode_steps'], 32)
        finally:
            O.M = original

    def test_startup_smoke_rejects_parallel_model_and_native_protocol_drift(self):
        original = O.M
        try:
            O.M = types.SimpleNamespace(validate_config=lambda *args: None)
            cases = (
                (('cluster', 'component_placement', 'actor'), '4'),
                (('env', 'train', 'total_num_envs'), 16),
                (('env', 'train', 'group_size'), 4),
                (('env', 'train', 'chunk'), 16),
                (('actor', 'micro_batch_size'), 4),
                (('rollout', 'model', 'kind'), 'different'),
                (('algorithm', 'update_epoch'), 1),
                (('env', 'eval', 'total_num_envs'), 16),
                (('env', 'eval', 'task_config', 'camera', 'collect_wrist_camera'), False),
                (('env', 'eval', 'max_episode_steps'), 384),
                (('runner', 'val_check_interval'), -1),
                (('actor', 'global_batch_size'), 512),
            )
            for path, value in cases:
                with self.subTest(path=path):
                    formal, startup, row = startup_fixture()
                    parent = startup
                    for key in path[:-1]:
                        parent = parent[key]
                    parent[path[-1]] = value
                    with self.assertRaises(AssertionError):
                        O.validate_startup_smoke(startup, row, {}, Path('/data/chenyiteng/formal-v1'), formal)
        finally:
            O.M = original

    def test_trial_order_and_unique_execution_identities(self):
        formal = dict(key='formal', namespace='opendw_formal', config='/data/chenyiteng/formal.yaml')
        _, _, startup = startup_fixture()
        self.assertEqual(O.trial_keys(dict(trials=[formal])), ['formal'])
        plan = dict(startup_smoke=True, trials=[startup, formal])
        self.assertEqual(O.trial_keys(plan), ['startup_smoke', 'formal'])
        invalid = [dict(startup_smoke=True, trials=[formal, startup]),
                   dict(startup_smoke=False, trials=[startup, formal]),
                   dict(startup_smoke=True, trials=[formal]),
                   dict(startup_smoke='yes', trials=[startup, formal])]
        for field in ('namespace', 'config'):
            duplicate = copy.deepcopy(plan)
            duplicate['trials'][0][field] = duplicate['trials'][1][field]
            invalid.append(duplicate)
        for value in invalid:
            with self.subTest(plan=value):
                with self.assertRaises(AssertionError):
                    O.trial_keys(value)

    def test_effective_200_budget_and_unchanged_training(self):
        smoke, control = configs()
        original = copy.deepcopy(smoke)
        cfg, manifest = build(smoke, control)
        self.assertEqual(smoke, original)
        self.assertEqual(O.protocol(cfg), O.protocol(smoke))
        self.assertEqual((cfg['runner']['max_epochs'], cfg['runner']['max_steps']), (1000, 200))
        self.assertEqual(cfg['runner']['save_interval'], cfg['runner']['val_check_interval'])
        self.assertEqual(manifest['budget']['scheduled_optimizer_steps_per_iteration'], 6)
        self.assertIsNone(cfg['runner']['resume_dir'])

    def test_native_eval_is_distinct_real_clean_aloha_three_view(self):
        cfg, _ = build(*configs())
        self.assertEqual(cfg['env']['train']['env_type'], 'opendw_robotwin')
        e = cfg['env']['eval']
        self.assertEqual(e['env_type'], 'robotwin')
        self.assertEqual(e['task_config']['task_name'], 'adjust_bottle')
        self.assertEqual(e['task_config']['embodiment'], ['aloha-agilex'])
        self.assertTrue(e['task_config']['camera']['collect_wrist_camera'])
        self.assertTrue(e['enable_offload'])
        self.assertEqual((e['max_episode_steps'], e['max_steps_per_rollout_epoch'], e['task_config']['step_lim']), (384, 384, 384))
        self.assertEqual(e['total_num_envs'], 32)
        self.assertNotIn('service_urls', e)

    def test_wrong_smoke_or_changed_control_budget_rejected(self):
        smoke, control = configs()
        smoke['env']['train']['max_episode_steps'] = 32
        with self.assertRaises(AssertionError):
            build(smoke, control)
        smoke, control = configs()
        control['runner']['max_steps'] = 1000
        with self.assertRaises(AssertionError):
            build(smoke, control)

    def test_loopback_service_alias_same_port_rejected(self):
        smoke, control = configs()
        with self.assertRaises(ValueError):
            B.build(smoke, control, name='formal-v1', run_dir='/data/chenyiteng/formal-v1',
                    services=['http://127.0.0.1:8126', 'http://127.0.0.1:8126/'],
                    native_assets='/data/chenyiteng/robotwin', native_eval_seeds='/data/chenyiteng/seeds.json')

    def test_direct_start_uses_frozen_protocol_without_learning_evidence(self):
        original = O.M
        try:
            O.M = types.SimpleNamespace(owned_path=lambda p: Path(p), H=types.SimpleNamespace(config=O.read))
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'full.json'
                path.write_text(json.dumps(configs()[0]))
                plan = dict(start_mode=O.DIRECT_START, protocol_reference=dict(config=str(path), sha256=O.sha(path)))
                self.assertEqual(O.protocol_reference(plan)['env']['train']['total_num_envs'], 64)
                path.write_text('{}')
                with self.assertRaises(AssertionError):
                    O.protocol_reference(plan)
        finally:
            O.M = original

    def test_adoption_requires_lifecycle_provenance_validator(self):
        original = O.M
        try:
            with tempfile.TemporaryDirectory() as temporary:
                cycle = Path(temporary)
                (cycle / 'rlt-stopped.json').write_text('{}')
                receipt = cycle / 'adoption.json'
                receipt.write_text('{}')
                calls = []
                O.M = types.SimpleNamespace(owned_path=lambda p: Path(p),
                    C=types.SimpleNamespace(verify_adoption=lambda stage, proof: calls.append((stage, proof))))
                plan = dict(start_mode=O.DIRECT_START, lifecycle_path=str(cycle),
                            borrow_adoption=dict(receipt=str(receipt), sha256=O.sha(receipt)))
                self.assertTrue(O.verify_adoption(plan))
                self.assertEqual(calls, [(cycle, receipt)])
                del plan['borrow_adoption']
                with self.assertRaises(AssertionError):
                    O.verify_adoption(plan)
        finally:
            O.M = original

    def learning_fixture(self, root):
        def write(name, data):
            path = root / name
            path.write_text(json.dumps(data))
            return path
        cfg = write('smoke.json', configs()[0])
        final = write('final.json', dict(mode='multigpu_smoke', terminal_status='completed', error=None,
                     recovery_error=dict(reason='return pending example'), trials=[dict(key='n64_full', exit_code=0)]))
        ranks = []
        for rank in (0, 1):
            p = root / f'rank{rank}.pt'
            p.write_bytes(b'CPU parser fixture; not a model')
            ranks.append(dict(rank=rank, stable=True, model=dict(finite=True), optimizer=dict(finite=True),
                              positive_lr_nonzero_adam_moments=True, adam_step_min=6, adam_step_max=6,
                              path=str(p), bytes=p.stat().st_size))
        row = dict(key='n64_full', learning_status='effective_update_verified', binding_verified=True, exit_code=0,
                   reconstruction=dict(status='reconstructed', trajectories=512, maximum_chunks=6144,
                     unknown_groups=0, retained_groups=2, nonzero_advantage_groups=1, valid_chunks_after_filter=88),
                   tensorboard=dict(finite=True, stable=True, nonzero_grad=True, nonzero_advantage=True),
                   checkpoint=dict(status='checked', finite_nonzero_update_state=True, cuda_initialized=False, ranks=ranks),
                   resources=dict(owned_outside_gpu4_7_contexts=[]))
        report = dict(analysis_kind='opendw_multigpu_smoke_analysis_v1', trials=[row], source_sha256={str(cfg): O.sha(cfg)})
        report_path = write('audit.json', report)
        gate = dict(path=str(report_path), sha256=O.sha(report_path), trial_key='n64_full',
                    owner_final_path=str(final), owner_final_sha256=O.sha(final),
                    smoke_config=str(cfg), smoke_config_sha256=O.sha(cfg))
        return dict(learning_evidence=gate), report, report_path

    def test_learning_gate_rejects_exit0_zero_signal_and_wrong_steps(self):
        original = O.M
        try:
            O.M = types.SimpleNamespace(owned_path=lambda p: Path(p), H=types.SimpleNamespace(config=O.read))
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for mutation in ('status', 'grad', 'adam_steps'):
                    with self.subTest(mutation=mutation):
                        plan, report, path = self.learning_fixture(root)
                        row = report['trials'][0]
                        if mutation == 'status':
                            row['learning_status'] = 'no_valid_group'
                        elif mutation == 'grad':
                            row['tensorboard']['nonzero_grad'] = False
                        else:
                            row['checkpoint']['ranks'][1]['adam_step_max'] = 2
                        path.write_text(json.dumps(report))
                        plan['learning_evidence']['sha256'] = O.sha(path)
                        with self.assertRaises(AssertionError):
                            O.learning_gate(plan)
                plan, _, _ = self.learning_fixture(root)
                self.assertEqual(O.learning_gate(plan)['env']['train']['total_num_envs'], 64)
        finally:
            O.M = original


if __name__ == '__main__':
    unittest.main()
