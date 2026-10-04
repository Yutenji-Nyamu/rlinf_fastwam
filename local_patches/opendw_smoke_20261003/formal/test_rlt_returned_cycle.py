"""Server CPU fixtures for returned-RLT provenance and graphics-scope handoff.

Run on SZ3 with the RLT Python; no Ray/GPU imports or live process signals.
Keep the sibling multigpu/rlt_gpu567_cycle.py source beside this formal folder.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).parent
C = module('returned_cycle_test', HERE / 'rlt_returned_cycle.py')
M = module('returned_combined_test', HERE / 'rlt_returned_multigpu_cycle.py')
BASE = HERE.parent / 'multigpu/rlt_gpu567_cycle.py'


class ReturnedCycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stage = self.root / 'new-cycle'
        self.stage.mkdir()
        self.helper = types.SimpleNamespace(same=Mock(return_value=False), now=lambda: 'fixture',
            MASKS={'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'}, read=C.read, save=self.save)
        self.addCleanup(patch.stopall)
        patch.object(C, 'H', self.helper).start()
        patch.object(C, 'owned', side_effect=lambda path, exists=True: Path(path)).start()

    def save(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def parent_fixture(self):
        owner, combined, child = [self.root / name for name in ('owner', 'combined', 'prior-gpu4')]
        child.mkdir()
        script = child / 'old-cycle.py'
        script.write_text('# frozen fixture')
        self.save(child / 'plan.json', {'runs': {'gpu4': {'new_run': '/rlt4/returned'}}})
        self.save(child / 'resumed-dispatched.json', {'cycle_id': child.name,
            'runs': {'gpu4': {'run': '/rlt4/returned'}}})
        self.save(combined / 'plan.json', {'children': {'gpu4': {'path': str(child),
            'module': str(script), 'plan_sha256': C.sha(child / 'plan.json'), 'module_sha256': C.sha(script)}}})
        self.save(combined / 'resumed-dispatched.json', {'children': {
            'gpu4': {'cycle_id': child.name}, 'gpu567': {'cycle_id': 'prior-gpu567'}}})
        self.save(owner / 'owner-plan.json', {'owner_dir': str(owner), 'lifecycle_path': str(combined)})
        self.save(owner / 'final.json', {'terminal_status': 'completed', 'error': None,
            'recovery_error': None, 'rlt_borrowed': True, 'rlt_return_dispatched': True,
            'rlt_first_round_verified': False})
        self.save(owner / 'owner-identity.json', {'pid': 101, 'uid': 20001, 'start': 123})
        self.save(owner / 'cleanup.json', {'all_stopped': True})
        self.save(owner / 'smoke-release.json', {'cycle_id': combined.name, 'gpus': [4, 5, 6, 7],
            'all_workers_stopped': True, 'terminal_status': 'completed',
            'managed_processes': [{'pid': 104, 'uid': 20001, 'start': 124}]})
        self.save(combined / 'return-started.json', {'release_sha256': C.sha(owner / 'smoke-release.json')})
        self.save(owner / 'rlt-return-dispatched.json', {
            'result': C.read(combined / 'resumed-dispatched.json')['children']})
        return owner, combined, child, script

    def test_parent_provenance_allows_pending_first_round_but_rejects_live_or_incomplete_owner(self):
        owner, combined, child, script = self.parent_fixture()
        group, identity, evidence = C.parent_complete(owner, child, script, self.helper)
        self.assertEqual(group, 'gpu4')
        self.assertEqual(evidence['child_return_sha256'], C.sha(child / 'resumed-dispatched.json'))
        self.helper.same.return_value = True
        with self.assertRaisesRegex(AssertionError, 'still live'):
            C.parent_complete(owner, child, script, self.helper)
        self.helper.same.return_value = False
        good = C.read(owner / 'final.json')
        for key, value in [('terminal_status', 'failed'), ('recovery_error', 'return failed'),
                           ('rlt_borrowed', False), ('rlt_return_dispatched', False)]:
            with self.subTest(key=key):
                self.save(owner / 'final.json', dict(good, **{key: value}))
                with self.assertRaises(AssertionError):
                    C.parent_complete(owner, child, script, self.helper)
        self.save(owner / 'final.json', good)
        self.save(child / 'resumed-dispatched.json', {'cycle_id': child.name,
            'runs': {'gpu4': {'run': '/unrelated/run'}}})
        with self.assertRaises(AssertionError):
            C.parent_complete(owner, child, script, self.helper)

    def test_failed_parent_requires_complete_cleanup_release_and_return(self):
        owner, combined, child, script = self.parent_fixture()
        final = C.read(owner / 'final.json')
        self.save(owner / 'final.json', dict(final, terminal_status='failed',
            error={'type': 'PermissionError', 'error': 'Unreadable unrelated process environment'}))
        release = dict(C.read(owner / 'smoke-release.json'), terminal_status='failed')
        self.save(owner / 'smoke-release.json', release)
        self.save(combined / 'return-started.json', {'release_sha256': C.sha(owner / 'smoke-release.json')})
        group, _, evidence = C.parent_complete(owner, child, script, self.helper)
        self.assertEqual(group, 'gpu4')
        self.assertEqual(evidence['release_sha256'], C.sha(owner / 'smoke-release.json'))
        self.helper.same.side_effect = lambda row: row['pid'] == 104
        with self.assertRaisesRegex(AssertionError, 'worker is still live'):
            C.parent_complete(owner, child, script, self.helper)
        self.helper.same.side_effect = None
        corruptions = [
            (owner / 'cleanup.json', {'all_stopped': False}),
            (owner / 'smoke-release.json', dict(release, all_workers_stopped=False)),
            (owner / 'smoke-release.json', dict(release, managed_processes=[])),
            (owner / 'smoke-release.json', dict(release, terminal_status='completed')),
            (combined / 'return-started.json', {'release_sha256': 'wrong'}),
            (owner / 'rlt-return-dispatched.json', {'result': {}}),
        ]
        for path, bad in corruptions:
            with self.subTest(path=path.name, value=bad):
                good = C.read(path)
                self.save(path, bad)
                with self.assertRaises(AssertionError):
                    C.parent_complete(owner, child, script, self.helper)
                self.save(path, good)

    def test_return_driver_is_bound_to_original_module_cycle_key_and_launch(self):
        prior = self.root / 'previous'
        script = prior / 'cycle.py'
        identity = {'pid': 102, 'uid': 20001, 'start': 456, 'namespace': 'returned-ns'}
        self.save(prior / 'plan.json', {'runs': {'gpu4': {'namespace': 'returned-ns', 'new_run': '/returned/4'}}})
        launched = {'identity': identity, 'namespace': 'returned-ns', 'run': '/returned/4'}
        self.save(prior / 'gpu4-launched.json', launched)
        argv = ['/python', str(script), '--cycle-dir', str(prior), 'driver', '--key', 'gpu4']

        def check(args):
            with patch.object(Path, 'read_bytes', return_value=b'\0'.join(x.encode() for x in args) + b'\0'):
                C.check_return_invocation(identity, prior, script, 'gpu4')

        check(argv)
        for index, value in [(1, '/other.py'), (3, '/wrong-cycle'), (4, 'status'), (6, 'gpu5')]:
            with self.subTest(index=index), self.assertRaises(AssertionError):
                altered = argv.copy()
                altered[index] = value
                check(altered)
        for field, value in [('start', 457), ('uid', 0), ('pid', 103)]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.save(prior / 'gpu4-launched.json', dict(launched, identity=dict(identity, **{field: value})))
                check(argv)

    def test_source_check_keeps_gpu4_and_gpu567_repository_heads_separate(self):
        for group in ('gpu4', 'gpu567'):
            with self.subTest(group=group):
                repo = self.root / group
                repo.mkdir()
                source = repo / 'source.py'
                source.write_text(group)
                plan = {'group': group, 'repo': str(repo), 'head': C.HEADS[group],
                        'source_sha256': {'source.py': C.sha(source)}}
                with patch('subprocess.check_output', side_effect=[C.HEADS[group] + '\n', '']) as git:
                    C.source_check(plan)
                self.assertTrue(all(call.args[0][2] == str(repo) for call in git.call_args_list))
                other = C.HEADS['gpu567' if group == 'gpu4' else 'gpu4']
                with patch('subprocess.check_output', return_value=other), self.assertRaises(AssertionError):
                    C.source_check(plan)

    def test_reused_base_binds_gpu4_stop_and_original_repository_without_signaling_owner(self):
        base = module('returned_stop_base_test', BASE)
        row = {'gpus': [4], 'original_run': '/rlt4/original', 'original_namespace': 'rlt4-ns',
               'original_identity': {'pid': 103, 'uid': 20001, 'start': 9}, 'original_jobs': []}
        pre = self.stage / 'prepared/gpu4'
        pre.mkdir(parents=True)
        (pre / 'original.yaml').write_text('fixture')
        row['prepared_sha256'] = {'original.yaml': C.sha(pre / 'original.yaml')}
        plan = {'group': 'gpu4', 'base_module': str(BASE), 'helper_path': '/helper.py',
                'runs': {'gpu4': row}, 'repo': '/separate-gpu4-repo', 'next_six_ops': '/ops.py',
                'old_owner': {'pid': 101, 'uid': 20001, 'start': 123}}
        self.save(self.stage / 'plan.json', plan)
        helper = self.helper
        helper.same = lambda ident: ident['pid'] == 103
        helper.actors = lambda _: []
        helper.active = lambda *_: []
        helper.validate_actor_rows = lambda *_: None
        helper.process_tree = lambda roots: {pid: {'pid': pid} for pid in roots}
        helper.gpu_processes = Mock(return_value=[])
        helper.config = lambda _: {'fixture': True}
        with patch.object(C, 'sha', return_value=C.BASE_SHA), patch.object(C, 'load', return_value=base), \
             patch.object(base, 'import_file', return_value=helper):
            C.install_helper(self.stage)
        self.assertEqual(base.KEY_GPUS, {'gpu4': [4]})
        self.assertEqual(helper.SCRIPT_NAME, C.SCRIPT_NAME)
        self.assertIs(helper.load_plan, C.load_plan)
        self.assertIs(helper.source_check, C.source_check)
        ops = types.SimpleNamespace()

        def stopped(arg):
            self.assertIsNone(arg)
            self.assertTrue((self.stage / 'monitor-retired.json').is_file())
            self.assertEqual(ops.checked()['old_runs'][0]['gpus'], [4])

        ops.stop_old = stopped
        with patch.object(base, 'load_plan', return_value=plan), patch.object(base, 'watch_targets'), \
             patch.object(base, 'read', return_value={}), patch.object(base, 'require_checkpoint') as cp, \
             patch.object(base, 'import_file', return_value=ops), \
             patch.object(base, 'finalize_stopped', return_value={'stopped': True}):
            self.assertTrue(C.stop(self.stage)['stopped'])
        cp.assert_called_once_with('/rlt4/original', {'fixture': True}, '/separate-gpu4-repo')
        self.assertIn('no signal sent', C.read(self.stage / 'monitor-retired.json')['reason'])

    def scope_fixture(self):
        old, new, runtime = [self.root / name for name in ('old-manifest.json', 'manifest.json', 'scope.py')]
        old.write_text('{}')
        new.write_text('{}')
        runtime.write_text('# fixture')
        activation = self.root / 'activation.json'
        plan = {'runs': {'gpu4': {'new_run': str(self.root / 'returned4')}},
                'scope_activation': str(activation), 'scope_id': 'token', 'scope_manifest': str(new),
                'legacy_scopes': {str(old): {'manifest_sha256': C.sha(old),
                    'bootstrap_dir': '/old/bootstrap', 'marker_path': '/old/libscope.so'}}}
        self.save(self.stage / 'plan.json', plan)
        fragment = {'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST': str(new),
                    'PYTHONPATH': '/new/bootstrap:/wm/repo', 'LD_PRELOAD': '/new/libscope.so',
                    '__GL_APPLICATION_PROFILE': '1', '__GL_APPLICATION_PROFILE_LOG': '0',
                    'HOME': '/home/chenyiteng'}
        receipt = {'status': 'active', 'uid': 20001, 'scope_id': 'token',
                   'environment_fragment': fragment, 'manifest_sha256': C.sha(new),
                   'runtime_path': str(runtime), 'runtime_sha256': C.sha(runtime)}
        manifest = {'bootstrap_path': '/new/bootstrap/sitecustomize.py', 'marker_path': '/new/libscope.so'}
        env = {'RLINF_OPENDW_GPU_SCOPE_MANIFEST': str(old), 'PYTHONPATH': '/old/bootstrap:/rlt/gpu4:/libs',
               'LD_PRELOAD': '/old/libscope.so:/unrelated.so', 'HOME': '/home/chenyiteng'}
        return activation, receipt, manifest, env

    def test_overlay_requires_activation_and_exact_environment_path(self):
        activation, receipt, manifest, env = self.scope_fixture()
        prepared = self.stage / 'prepared/gpu4/environment.json'
        self.assertIs(C.scope_overlay(self.stage, prepared, env), env)
        self.save(activation, receipt)
        with patch.object(C, 'load') as loaded:
            self.assertIs(C.scope_overlay(self.stage, self.root / 'unrelated/environment.json', env), env)
            loaded.assert_not_called()
        self.save(activation, dict(receipt, status='rolled_back'))
        self.assertIs(C.scope_overlay(self.stage, prepared, env), env)

    def test_overlay_replaces_exact_legacy_prefix_and_preserves_each_rlt_path(self):
        activation, receipt, manifest, env = self.scope_fixture()
        self.save(activation, receipt)
        scope = types.SimpleNamespace(read_manifest=Mock(return_value=manifest))
        for tail in ('/rlt/gpu4', '/rlt/gpu567'):
            with self.subTest(repo=tail), patch.object(C, 'load', return_value=scope):
                original = dict(env, PYTHONPATH='/old/bootstrap:' + tail + ':/libs')
                result = C.scope_overlay(self.stage, self.stage / 'prepared/gpu4/environment.json', original)
                self.assertEqual(result['PYTHONPATH'], '/new/bootstrap:' + tail + ':/libs')
                self.assertEqual(result['LD_PRELOAD'], '/new/libscope.so:/unrelated.so')
                self.assertEqual(result['__GL_APPLICATION_PROFILE_LOG'], '0')
                self.assertNotIn('RLINF_OPENDW_GPU_SCOPE_MANIFEST', result)
                self.assertIn('RLINF_OPENDW_GPU_SCOPE_MANIFEST', original)
                self.assertNotIn('/wm/repo', result['PYTHONPATH'])
                # The runtime environment is read again inside driver; overlay is idempotent.
                self.assertEqual(C.scope_overlay(self.stage, self.root / 'returned4/runtime/environment.json', result), result)
        self.save(activation, dict(receipt, uid=0))
        with self.assertRaises(AssertionError):
            C.scope_overlay(self.stage, self.stage / 'prepared/gpu4/environment.json', env)
        bad = copy.deepcopy(receipt)
        bad['environment_fragment']['CUDA_VISIBLE_DEVICES'] = '4'
        self.save(activation, bad)
        with self.assertRaises(AssertionError):
            C.scope_overlay(self.stage, self.stage / 'prepared/gpu4/environment.json', env)

    def test_overlay_rejects_changed_legacy_manifest(self):
        activation, receipt, manifest, env = self.scope_fixture()
        self.save(activation, receipt)
        Path(env['RLINF_OPENDW_GPU_SCOPE_MANIFEST']).write_text('{"changed": true}')
        with patch.object(C, 'load', return_value=types.SimpleNamespace(read_manifest=lambda _: manifest)), \
             self.assertRaises(AssertionError):
            C.scope_overlay(self.stage, self.stage / 'prepared/gpu4/environment.json', env)

    def test_new_combined_partial_return_recognizes_gpu4_base_markers(self):
        gpu4, gpu567 = self.root / 'gpu4', self.root / 'gpu567'
        gpu4.mkdir()
        gpu567.mkdir()
        (gpu4 / 'old-stopped.json').write_text('{}')
        (self.stage / 'clean-old-stop-attempt.json').write_text('{}')
        plan = {'children': {'gpu4': {'path': str(gpu4)}, 'gpu567': {'path': str(gpu567)}}}
        release = self.root / 'release.json'
        self.save(release, {'cycle_id': self.stage.name, 'all_workers_stopped': True,
                           'gpus': [4, 5, 6, 7], 'managed_processes': []})
        first = types.SimpleNamespace(finalize_stopped=Mock(side_effect=lambda stage:
            self.save(stage / 'rlt-stopped.json', {})), H=types.SimpleNamespace(resume=Mock(return_value={'returned': True})))
        sibling = types.SimpleNamespace(load_plan=lambda _: {'runs': {'gpu5': {'original_identity': {}}}},
            H=types.SimpleNamespace(same=Mock(return_value=True), resume=Mock()), finalize_stopped=Mock())
        with patch.object(M, 'load_plan', return_value=plan), patch.object(M, 'check_path', side_effect=Path), \
             patch.object(M, 'H', self.helper), patch.object(M, 'CHILDREN', {'gpu4': first, 'gpu567': sibling}):
            result = M.recover_partial(self.stage, release)
        self.assertEqual(M.CHILD_ATTEMPT_MARKERS['gpu4'], 'old-stop-attempt.json')
        first.finalize_stopped.assert_called_once_with(gpu4)
        first.H.resume.assert_called_once()
        sibling.finalize_stopped.assert_not_called()
        sibling.H.resume.assert_not_called()
        self.assertTrue(result['gpu567']['original_drivers_still_running'])

    def adopted_fixture(self):
        owner, combined, child, script = self.parent_fixture()
        (child / 'resumed-dispatched.json').unlink()
        (combined / 'resumed-dispatched.json').unlink()
        (combined / 'return-started.json').unlink()
        (owner / 'rlt-return-dispatched.json').unlink()
        sibling = self.root / 'prior-gpu567'
        sibling.mkdir()
        other_script = sibling / 'old-cycle.py'
        other_script.write_text('# sibling fixture')
        self.save(sibling / 'plan.json', {'runs': {f'gpu{g}': {} for g in (5, 6, 7)}})
        combined_plan = C.read(combined / 'plan.json')
        combined_plan['children']['gpu567'] = {'path': str(sibling), 'module': str(other_script),
            'plan_sha256': C.sha(sibling / 'plan.json'), 'module_sha256': C.sha(other_script)}
        self.save(combined / 'plan.json', combined_plan)
        for path, gpus in ((child, [4]), (sibling, [5, 6, 7]), (combined, [4, 5, 6, 7])):
            self.save(path / 'rlt-stopped.json', {'cycle_id': path.name, 'gpus_released': gpus,
                'all_original_drivers_stopped': True, 'all_original_namespaces_empty': True,
                'runs': {f'gpu{g}': {} for g in gpus}})
        final = {'terminal_status': 'failed', 'error': {'type': 'RuntimeError', 'error': 'Owner received signal 15'},
            'recovery_error': {'type': 'BlockingIOError', 'error': '[Errno 11] Resource temporarily unavailable'},
            'rlt_borrowed': True, 'rlt_return_dispatched': False}
        self.save(owner / 'final.json', final)
        self.save(owner / 'cleanup.json', {'all_stopped': True})
        self.save(owner / 'smoke-release.json', {'cycle_id': combined.name, 'gpus': [4, 5, 6, 7],
            'all_workers_stopped': True, 'terminal_status': 'failed', 'managed_processes': []})
        intent = self.root / 'handoff-intent.json'
        self.save(intent, {'kind': 'opendw-formal-adopt-held-rlt', 'parent_owner': str(owner),
            'lifecycle_path': str(combined), 'physical_gpus': [4, 5, 6, 7],
            'owner_plan_sha256': C.sha(owner / 'owner-plan.json'),
            'combined_plan_sha256': C.sha(combined / 'plan.json'),
            'owner_identity': C.read(owner / 'owner-identity.json'),
            'holder_identity': {'pid': C.os.getpid(), 'uid': 20001, 'start': 789},
            'child_plan_sha256': {key: row['plan_sha256'] for key,row in combined_plan['children'].items()},
            'child_module_sha256': {key: row['module_sha256'] for key,row in combined_plan['children'].items()}})
        return owner, combined, child, script, intent

    def test_adoption_accepts_exact_interruption_and_clean_completion_race(self):
        owner, combined, child, script, intent = self.adopted_fixture()
        self.assertEqual(C.adopted_parent(owner, child, script, intent)[0], 'gpu4')
        final = C.read(owner / 'final.json')
        self.save(owner / 'final.json', dict(final, terminal_status='completed', error=None))
        release = C.read(owner / 'smoke-release.json')
        self.save(owner / 'smoke-release.json', dict(release, terminal_status='completed'))
        self.assertEqual(C.adopted_parent(owner, child, script, intent)[0], 'gpu4')

    def test_adoption_rejects_arbitrary_failure_or_any_return_launch(self):
        owner, combined, child, script, intent = self.adopted_fixture()
        final = C.read(owner / 'final.json')
        self.save(owner / 'final.json', dict(final, error={'type': 'RuntimeError', 'error': 'unrelated crash'}))
        with self.assertRaises(AssertionError):
            C.adopted_parent(owner, child, script, intent)
        self.save(owner / 'final.json', final)
        (child / 'gpu4-launch-attempt.json').write_text('{}')
        with self.assertRaises(AssertionError):
            C.adopted_parent(owner, child, script, intent)

    def test_adoption_requires_current_live_holder_and_still_held_operation_lock(self):
        owner, combined, child, script, intent = self.adopted_fixture()
        self.helper.same = lambda ident: ident['pid'] == C.os.getpid()
        self.helper.gpu_processes = Mock(return_value=[])
        with self.assertRaisesRegex(AssertionError, 'retain.*lock'):
            C.adopted_parent(owner, child, script, intent, self.helper)
        with (combined / 'operation.lock').open('a') as held:
            C.fcntl.flock(held, C.fcntl.LOCK_EX | C.fcntl.LOCK_NB)
            self.assertEqual(C.adopted_parent(owner, child, script, intent, self.helper)[0], 'gpu4')
            C.fcntl.flock(held, C.fcntl.LOCK_UN)

    def test_successor_verifies_adoption_without_old_holder_pid_and_rejects_wrong_receipt(self):
        owner, combined, previous, script, intent = self.adopted_fixture()
        children, receipts = {}, {}
        original_children = C.read(combined/'plan.json')['children']
        for key, gpus in M.CHILD_GPUS.items():
            child = self.root/('new-'+key)
            source = original_children[key]['path']
            self.save(child/'plan.json', {'runs': {f'gpu{g}': {'original_identity': {'pid': 100+g}}
                                                 for g in gpus}})
            self.save(child/'rlt-stopped.json', {'cycle_id': child.name, 'gpus_released': gpus,
                'all_original_drivers_stopped': True, 'all_original_namespaces_empty': True,
                'runs': {f'gpu{g}': {} for g in gpus}})
            self.save(child/'adopted.json', {'cycle_id': child.name, 'physical_gpus': gpus,
                'rlt_remained_stopped': True, 'adopted_from': {'cycle': source,
                    'handoff_intent': str(intent), 'handoff_intent_sha256': C.sha(intent)}})
            children[key] = {'path': str(child)}
            receipts[key] = {'path': str(child/'adopted.json'), 'sha256': C.sha(child/'adopted.json')}
        runs = {f'gpu{g}': {} for g in (4, 5, 6, 7)}
        plan = {'children': children, 'runs': runs}
        self.save(self.stage/'rlt-stopped.json', {'cycle_id': self.stage.name, 'gpus_released': [4, 5, 6, 7],
            'all_original_drivers_stopped': True, 'all_original_namespaces_empty': True, 'runs': runs})
        receipt = {'kind': 'opendw-formal-adopt-held-rlt', 'cycle_id': self.stage.name,
            'physical_gpus': [4, 5, 6, 7], 'rlt_remained_stopped': True, 'handoff_intent': str(intent),
            'handoff_intent_sha256': C.sha(intent), 'parent_owner': str(owner),
            'parent_release': str(owner/'smoke-release.json'),
            'parent_release_sha256': C.sha(owner/'smoke-release.json'), 'child_receipts': receipts}
        self.save(self.stage/'adopted.json', receipt)
        with patch.object(M, 'load_plan', return_value=plan), patch.object(M, 'check_path', side_effect=Path), \
             patch.object(M, 'H', self.helper):
            self.assertEqual(M.verify_adoption(self.stage), receipt)
            self.save(self.stage/'adopted.json', dict(receipt, cycle_id='wrong-cycle'))
            with self.assertRaises(AssertionError):
                M.verify_adoption(self.stage)


if __name__ == '__main__':
    unittest.main(verbosity=2)
