"""CPU-only composition/receipt fixtures; never imports real lifecycle helpers."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('rlt_multigpu_cycle.py')
spec = importlib.util.spec_from_file_location('rlt_multigpu_under_test', SOURCE)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream)


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage = self.root/'combined'
        self.stage.mkdir()
        self.calls, self.components, self.subplans = [], {}, {}
        self.plan = {'cycle_id': self.stage.name, 'physical_gpus': [4, 5, 6, 7],
                     'python': '/python', 'ray_address': 'auto', 'ray_dashboard_url': 'http://127.0.0.1:8265',
                     'children': {}, 'runs': {}}
        for key, gpus in C.CHILD_GPUS.items():
            child = self.root/key
            child.mkdir()
            module = child/'component.py'
            module.write_text('# fixture only\n')
            sub = {k: self.plan[k] for k in ('python', 'ray_address', 'ray_dashboard_url')}
            sub['runs'] = {f'gpu{g}': {'gpus': [g], 'original_identity': {'pid': 1000+g, 'uid': 20001, 'start': g}}
                           for g in gpus}
            save(child/'plan.json', sub)
            self.subplans[key] = sub
            self.plan['runs'].update(sub['runs'])
            self.plan['children'][key] = {'path': str(child), 'module': str(module),
                                         'module_sha256': C.sha(module), 'plan_sha256': C.sha(child/'plan.json')}
            # Separate function globals imitate separately imported helper modules.
            scope = {'TAG': key, 'CALLS': self.calls, 'read': C.read, 'save': save}
            exec('def resume(stage, receipt):\n'
                 '    marker = stage / "fixture-dispatched.json"\n'
                 '    if not marker.exists():\n'
                 '        CALLS.append(("resume", TAG, read(receipt)))\n'
                 '        save(marker, {"tag": TAG})\n'
                 '    return {"tag": TAG}\n', scope)
            def state(stage, key=key, gpus=gpus):
                self.calls.append(('status', key))
                return {'runs': {f'gpu{g}': {'first_round_verified': True} for g in gpus}}
            helper = SimpleNamespace(resume=scope['resume'], status=state, same=lambda row: True,
                                     save=save, now=lambda: 'now', gpu_processes=lambda g: [])
            component = SimpleNamespace(H=helper, load_plan=lambda stage, sub=sub: copy.deepcopy(sub),
                                        install_helper=lambda stage: None)
            component.finalize_stopped = lambda stage, key=key: self.complete_child(key)
            self.components[key] = component
        self.helper = SimpleNamespace(now=lambda: 'now', save=save, same=lambda row: False,
                                      gpu_processes=lambda g: [])
        self.release_path = self.root/'release.json'
        self.release = {'cycle_id': self.stage.name, 'gpus': [4, 5, 6, 7], 'all_workers_stopped': True,
                        'terminal_status': 'failed', 'managed_processes': [{'pid': 999, 'uid': 20001, 'start': 7}]}
        save(self.release_path, self.release)
        self.addCleanup(patch.stopall)
        patch.object(C, 'H', self.helper).start()
        patch.object(C, 'CHILDREN', self.components).start()
        patch.object(C, 'check_path', side_effect=lambda path: Path(path)).start()
        self.load_patch = patch.object(C, 'load_plan', side_effect=lambda stage: copy.deepcopy(self.plan))
        self.load_patch.start()

    def complete_child(self, key):
        child = Path(self.plan['children'][key]['path'])
        receipt = {'cycle_id': child.name, 'gpus_released': C.CHILD_GPUS[key],
                   'runs': {k: {'recovery': {}} for k in self.subplans[key]['runs']},
                   'all_original_drivers_stopped': True, 'all_original_namespaces_empty': True}
        save(child/'rlt-stopped.json', receipt)
        return receipt

    def complete_all(self):
        for key in C.CHILD_GPUS:
            self.complete_child(key)
        return C.finalize_stopped(self.stage)

    def test_full_receipt_combines_exact_runs_and_child_releases_keep_subsets(self):
        receipt = self.complete_all()
        self.assertEqual(set(receipt['runs']), {'gpu4', 'gpu5', 'gpu6', 'gpu7'})
        C.resume(self.stage, self.release_path)
        C.resume(self.stage, self.release_path)
        calls = [c for c in self.calls if c[0] == 'resume']
        self.assertEqual([c[1] for c in calls], ['gpu4', 'gpu567'])
        for _, key, release in calls:
            self.assertEqual(release['gpus'], C.CHILD_GPUS[key])
            self.assertEqual(release['cycle_id'], Path(self.plan['children'][key]['path']).name)
        self.assertEqual(C.read(self.release_path)['gpus'], [4, 5, 6, 7])

    def test_live_owned_process_rejects_both_return_paths(self):
        self.complete_all()
        self.helper.same = lambda row: True
        with self.assertRaises(AssertionError):
            C.resume(self.stage, self.release_path)
        (self.stage/'rlt-stopped.json').unlink()
        save(self.stage/'clean-old-stop-attempt.json', {})
        with self.assertRaises(AssertionError):
            C.recover_partial(self.stage, self.release_path)
        self.assertFalse(any(c[0] == 'resume' for c in self.calls))

    def test_partial_leaves_unstopped_child_and_is_receipt_bound(self):
        save(self.stage/'clean-old-stop-attempt.json', {})
        self.complete_child('gpu4')
        result = C.recover_partial(self.stage, self.release_path)
        self.assertTrue(result['gpu567']['original_drivers_still_running'])
        self.assertTrue(result['gpu567']['return_not_needed'])
        self.assertEqual([c[1] for c in self.calls if c[0] == 'resume'], ['gpu4'])
        self.assertEqual(C.recover_partial(self.stage, self.release_path), result)
        self.assertFalse((self.stage/'rlt-stopped.json').exists())
        changed = dict(self.release, terminal_status='completed')
        self.release_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(AssertionError, 'receipt changed'):
            C.recover_partial(self.stage, self.release_path)

    def test_partial_uses_567_exact_old_stopped_marker(self):
        save(self.stage/'clean-old-stop-attempt.json', {})
        child = Path(self.plan['children']['gpu567']['path'])
        save(child/'old-stopped.json', {})
        result = C.recover_partial(self.stage, self.release_path)
        self.assertEqual(result['gpu567']['tag'], 'gpu567')
        self.assertTrue(result['gpu4']['return_not_needed'])
        self.assertEqual([c[1] for c in self.calls if c[0] == 'resume'], ['gpu567'])

    def test_ambiguous_partial_child_never_resumed(self):
        save(self.stage/'clean-old-stop-attempt.json', {})
        self.components['gpu567'].H.same = lambda row: False
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            C.recover_partial(self.stage, self.release_path)
        self.assertFalse(any(c[0] == 'resume' for c in self.calls))
        with self.assertRaisesRegex(AssertionError, 'previous partial'):
            C.recover_partial(self.stage, self.release_path)

    def test_facade_preserves_child_globals_and_uses_567_status_observer(self):
        original = self.components['gpu4'].H.resume
        mapping = {v['module']: self.components[k] for k, v in self.plan['children'].items()}
        def status567(stage):
            self.calls.append(('module_status', 'gpu567'))
            return {'runs': {f'gpu{g}': {'first_round_verified': True} for g in (5, 6, 7)},
                    'secondary_contexts': []}
        self.components['gpu567'].status = status567
        with patch.object(C, 'import_file', side_effect=lambda name, path: mapping[str(path)]):
            C.install_helper(self.stage)
        self.assertIs(self.components['gpu4'].H.resume, original)
        self.assertIs(C.H.resume, C.resume)
        self.assertEqual(original.__globals__['TAG'], 'gpu4')
        result = C.H.status(self.stage)
        self.assertTrue(result['all_first_rounds_verified'])
        self.assertIn(('module_status', 'gpu567'), self.calls)

    def test_prepare_rejects_567_previous_stop_attempt(self):
        child = Path(self.plan['children']['gpu567']['path'])
        save(child/'old-stop-attempt.json', {})
        mapping = {v['module']: self.components[k] for k, v in self.plan['children'].items()}
        with patch.object(C, 'ROOT', self.root), patch.object(C, 'import_file', side_effect=lambda name, path: mapping[str(path)]):
            with self.assertRaises(AssertionError):
                C.prepare(self.root/'new-combined', self.plan['children'])
        self.assertFalse((self.root/'new-combined').exists())


if __name__ == '__main__':
    unittest.main()
