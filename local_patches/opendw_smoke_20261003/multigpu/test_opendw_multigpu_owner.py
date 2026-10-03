"""CPU-only contract/ordering checks. Run on the server; never starts Ray/GPU."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('opendw_multigpu_owner.py')
spec = importlib.util.spec_from_file_location('multigpu_owner_under_test', SOURCE)
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)


def contract(owner):
    model = {'num_action_chunks': 32, 'openpi': {'action_chunk': 32, 'action_horizon': 50}}
    urls = ['http://127.0.0.1:18946', 'http://127.0.0.1:18947']
    cfg = {
        'cluster': {'component_placement': dict(O.PLACEMENT)},
        'runner': {'max_steps': 1, 'max_epochs': 1, 'val_check_interval': -1, 'save_interval': 1,
                   'only_eval': False, 'resume_dir': None, 'ckpt_path': None,
                   'logger': {'log_path': str(owner/'n64'/'logs')}},
        'env': {'enable_offload': True, 'train': {
            'env_type': 'opendw_robotwin', 'task_name': 'adjust_bottle', 'total_num_envs': 64,
            'rollout_epoch': 8, 'group_size': 8, 'chunk': 32, 'max_episode_steps': 32,
            'max_steps_per_rollout_epoch': 32, 'frame_stride': 4, 'auto_reset': False,
            'ignore_terminations': False, 'use_rel_reward': True, 'reward_coef': 1.0,
            'success_reward_threshold': 0.9, 'service_urls': urls,
            'enable_offload': True, 'enable_init_offload': True}},
        'algorithm': {'group_size': 8, 'adv_type': 'grpo', 'reward_type': 'chunk_level',
                      'filter_rewards': True, 'rewards_lower_bound': 0.1, 'rewards_upper_bound': 0.9,
                      'update_epoch': 2},
        'actor': {'global_batch_size': 512, 'micro_batch_size': 8, 'model': model, 'enable_offload': True},
        'rollout': {'model': copy.deepcopy(model), 'pipeline_stage_num': 1, 'enable_offload': True}}
    row = {'key': 'n64', 'num_envs': 64, 'episode_steps': 32, 'timeout_seconds': 3600}
    plan = {'services': [{'key': 'wm6', 'physical_gpu': 6, 'url': urls[0]},
                         {'key': 'wm7', 'physical_gpu': 7, 'url': urls[1]}]}
    return cfg, row, plan


class ContractTests(unittest.TestCase):
    def test_exact_scale_and_reject_budget_route_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = Path(directory)
            cfg, row, plan = contract(owner)
            O.validate_config(cfg, row, plan, owner)
            full = copy.deepcopy(cfg)
            full['env']['train'].update(max_episode_steps=384, max_steps_per_rollout_epoch=384)
            full['actor']['global_batch_size'] = 2048
            O.validate_config(full, dict(row, episode_steps=384), plan, owner)
            with self.assertRaises(AssertionError):
                O.validate_config(full, row, plan, owner)
            for path, value in [
                ('cluster.component_placement.env', '4,5'), ('env.train.rollout_epoch', 1),
                ('env.train.total_num_envs', 16), ('env.train.max_episode_steps', 384),
                ('actor.global_batch_size', 64), ('actor.micro_batch_size', 16),
                ('algorithm.update_epoch', 1), ('runner.max_steps', 2),
                ('runner.val_check_interval', 10), ('env.train.service_urls', list(reversed(cfg['env']['train']['service_urls']))),
                ('runner.logger.log_path', str(owner/'different-trial')),
            ]:
                with self.subTest(path=path):
                    changed = copy.deepcopy(cfg)
                    node = changed
                    parts = path.split('.')
                    for part in parts[:-1]:
                        node = node[part]
                    node[parts[-1]] = value
                    with self.assertRaises(AssertionError):
                        O.validate_config(changed, row, plan, owner)

    def test_suite_order_is_exact_short_then_full(self):
        rows = [{'key': 'short', 'namespace': 'opendw_short', 'episode_steps': 32},
                {'key': 'full', 'namespace': 'opendw_full', 'episode_steps': 384}]
        O.validate_trial_order(rows)
        for invalid in (list(reversed(rows)), rows[:1], rows + rows[:1]):
            with self.assertRaises(AssertionError):
                O.validate_trial_order(invalid)

    def test_managed_gpu_and_per_service_gpu_boundaries(self):
        class Catalog:
            def scan(self):
                pass
            def live(self, phase=None):
                rows = [{'pid': 999991, 'phase': 'service_wm6'}, {'pid': 999992, 'phase': 'n64'}]
                return [r for r in rows if phase is None or r['phase'] == phase]
        with tempfile.TemporaryDirectory() as directory:
            owner = Path(directory)
            _, _, plan = contract(owner)
            plan['owner_dir'] = str(owner)
            for gpu_rows, passes in [
                ([{'pid': 999991, 'gpu': 6}, {'pid': 999992, 'gpu': 4}, {'pid': 777777, 'gpu': 0}], True),
                ([{'pid': 999991, 'gpu': 7}], False),
                ([{'pid': 999992, 'gpu': 0}], False),
            ]:
                helper = SimpleNamespace(now=lambda: 'now', gpu_processes=lambda g: gpu_rows)
                with patch.object(O, 'H', helper), patch.object(O.subprocess, 'check_output', return_value=''):
                    if passes:
                        O.resource_snapshot(plan, Catalog(), 'trial')
                    else:
                        with self.assertRaises(AssertionError):
                            O.resource_snapshot(plan, Catalog(), 'trial')

    def test_lifecycle_import_fails_before_unreviewed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'unreviewed.py'
            path.write_text('raise RuntimeError("must not import")\n')
            plan = {'lifecycle_path': str(root), 'lifecycle_module': str(path), 'source_sha256': {str(path): '0'*64}}
            with patch.object(O, 'owned_path', side_effect=lambda value, exists=True: Path(value)):
                with self.assertRaisesRegex(AssertionError, 'Unfrozen lifecycle'):
                    O.load_lifecycle(plan)

    def test_reused_pid_never_signalled(self):
        helper = SimpleNamespace(same=lambda ident: False)
        with patch.object(O, 'H', helper), patch.object(O, 'pidfd_open') as opened:
            O.Catalog(Path('/unused'), 'token').send({'pid': 123, 'start': 456}, 15)
            opened.assert_not_called()


class SuiteOrderingTests(unittest.TestCase):
    def run_suite(self, fail_second_ready=False, partial_stop=False, full_release_busy=False, first_round_pending=False):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner, cycle, repo = root/'owner', root/'cycle', root/'repo'
            cycle.mkdir()
            repo.mkdir()
            (cycle/'plan.json').write_text('{}')
            environment = root/'env.json'
            environment.write_text('{}')
            cp = {'ray_address': 'auto', 'ray_dashboard_url': 'http://127.0.0.1:8265'}
            rows = [{'key': key, 'namespace': 'opendw_'+key, 'num_envs': 64, 'episode_steps': 32 if key == 'trial1' else 384,
                     'config': str(root/(key+'.json')), 'timeout_seconds': 30} for key in ('trial1', 'trial2')]
            for row in rows:
                Path(row['config']).write_text('{}')
            services = [{'key': 'wm'+str(g), 'physical_gpu': g, 'url': 'http://127.0.0.1:'+str(18000+g),
                         'argv': ['fake-python', 'service', str(g)], 'cwd': str(repo)} for g in (6, 7)]
            plan = {'owner_dir': str(owner), 'lifecycle_path': str(cycle), 'repo': str(repo),
                    'environment_file': str(environment), 'python': 'fake-python', 'trials': rows,
                    'services': services, 'mode': 'multigpu_smoke', 'restore_wait_seconds': 60}
            children = []
            def save(path, value):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(value))
            def stop(stage):
                events.append('borrow')
                if partial_stop:
                    save(stage/'clean-old-stop-attempt.json', {})
                    raise RuntimeError('Only one child completed its stop')
                save(stage/'rlt-stopped.json', {})
            def resume(stage, receipt):
                events.append('resume')
                self.assertTrue(Path(receipt).is_file())
            def gpu_processes(gpus):
                if gpus == O.GPUS:
                    events.append('check_all_four_empty')
                    return [{'gpu': 7, 'pid': 12345}] if full_release_busy else []
                return []
            def recover_partial(stage, receipt):
                events.append('recover_partial')
                value = json.loads(Path(receipt).read_text())
                self.assertEqual(value['gpus'], O.GPUS)
                self.assertTrue(value['all_workers_stopped'])
                self.assertNotIn('all_gpus_empty', value)
                self.assertFalse((stage/'rlt-stopped.json').exists())
                return {'status': 'returned_proven_child_only'}
            helper = SimpleNamespace(now=lambda: 'now', proc=lambda pid: {'pid': pid, 'uid': O.UID, 'start': pid},
                                     active=lambda rows, ns: [], actors=lambda p: [], atomic=save, save=save,
                                     gpu_processes=gpu_processes, resume=resume,
                                     status=lambda stage: {'all_first_rounds_verified': not first_round_pending,
                                         'runs': {f'gpu{g}': {} for g in (4, 5, 6, 7)}})
            lifecycle = SimpleNamespace(load_plan=lambda stage: cp, stop=stop, recover_partial=recover_partial)
            class Catalog:
                def __init__(self, *args):
                    self.rows = {}
                def add(self, ident, phase, proof):
                    self.rows[ident['pid']] = dict(ident, phase=phase)
                def live(self, phase=None):
                    return []
            class Child:
                def __init__(self, argv, **kwargs):
                    self.pid = 990000 + len(children)
                    self.key = argv[argv.index('--key')+1] if '--key' in argv else None
                    self.returncode = None
                    self.polls = 0
                    children.append(self)
                    if self.key:
                        events.append('start_'+self.key)
                        save(owner/self.key/'driver-finished.json', {'exit_code': 0})
                        save(owner/self.key/'verified-placement.json', {})
                def poll(self):
                    self.polls += 1
                    if self.key and self.polls > 1:
                        self.returncode = 0
                    return self.returncode
                def wait(self, timeout):
                    return 0
            def http(url, endpoint='/health', **kwargs):
                index = next(i for i, service in enumerate(services) if service['url'] == url)
                if endpoint == '/offload':
                    events.append('offload_'+services[index]['key'])
                    return {'ok': True, 'is_offloaded': True}
                events.append('ready_'+services[index]['key'])
                return {'ok': not (fail_second_ready and index == 1), 'is_offloaded': True,
                        'pid': children[index].pid, 'physical_gpu': services[index]['physical_gpu']}
            def cleanup(plan, catalog, phase=None):
                events.append('cleanup_'+str(phase))
                return {'all_stopped': True}
            tick = {'value': 0}
            def monotonic():
                tick['value'] += 61 if first_round_pending and 'resume' in events else 1
                return tick['value']
            with patch.object(O, 'validate', return_value=(owner, cycle, repo, {})), \
                 patch.object(O, 'C', lifecycle), patch.object(O, 'H', helper), \
                 patch.object(O, 'Catalog', Catalog), patch.object(O, 'add_allowlist'), \
                 patch.object(O.subprocess, 'Popen', Child), patch.object(O, 'http', http), \
                 patch.object(O, 'resource_snapshot'), patch.object(O, 'register_actors'), \
                 patch.object(O, 'cleanup', cleanup), patch.object(O.signal, 'signal'), \
                 patch.object(O.socket, 'socket'), patch.object(O.time, 'sleep'), \
                 patch.object(O.time, 'monotonic', side_effect=monotonic):
                if fail_second_ready or partial_stop or full_release_busy:
                    with self.assertRaises(SystemExit):
                        O.owner_main(plan)
                else:
                    O.owner_main(plan)
            return events, json.loads((owner/'final.json').read_text())

    def test_both_cpu_ready_then_borrow_and_single_return_after_suite(self):
        events, final = self.run_suite()
        self.assertLess(events.index('ready_wm6'), events.index('borrow'))
        self.assertLess(events.index('ready_wm7'), events.index('borrow'))
        self.assertLess(events.index('cleanup_trial1'), events.index('start_trial2'))
        self.assertLess(events.index('cleanup_trial2'), events.index('cleanup_None'))
        self.assertLess(events.index('cleanup_None'), events.index('resume'))
        self.assertLess(events.index('check_all_four_empty'), events.index('resume'))
        self.assertEqual(events.count('borrow'), 1)
        self.assertEqual(events.count('resume'), 1)
        self.assertEqual(final['terminal_status'], 'completed')
        self.assertTrue(final['rlt_first_round_verified'])

    def test_second_service_failed_readiness_never_borrows(self):
        events, final = self.run_suite(True)
        self.assertNotIn('borrow', events)
        self.assertNotIn('resume', events)
        self.assertEqual(final['terminal_status'], 'failed')
        self.assertFalse(final['rlt_borrowed'])

    def test_partial_stop_returns_only_via_explicit_partial_hook(self):
        events, final = self.run_suite(partial_stop=True)
        self.assertLess(events.index('cleanup_None'), events.index('recover_partial'))
        self.assertNotIn('resume', events)
        self.assertNotIn('check_all_four_empty', events)
        self.assertFalse(final['rlt_borrowed'])
        self.assertFalse(final['rlt_first_round_verified'])
        self.assertTrue(final['partial_rlt_recovery_recorded'])
        self.assertEqual(final['terminal_status'], 'failed')

    def test_full_release_requires_all_four_cards_empty(self):
        events, final = self.run_suite(full_release_busy=True)
        self.assertIn('check_all_four_empty', events)
        self.assertNotIn('resume', events)
        self.assertNotIn('recover_partial', events)
        self.assertIsNotNone(final['recovery_error'])

    def test_dispatched_slow_first_round_is_pending_not_wm_failure(self):
        events, final = self.run_suite(first_round_pending=True)
        self.assertIn('resume', events)
        self.assertEqual(final['terminal_status'], 'completed')
        self.assertIsNone(final['error'])
        self.assertIsNone(final['recovery_error'])
        self.assertTrue(final['rlt_return_dispatched'])
        self.assertTrue(final['rlt_first_round_pending'])
        self.assertFalse(final['rlt_first_round_verified'])


if __name__ == '__main__':
    unittest.main()
