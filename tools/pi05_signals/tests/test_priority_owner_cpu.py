"""CPU-only regression checks; fake process/owner objects, never /proc or GPU."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
import ast
import json
import re
from unittest.mock import patch

from priority_owner import borrowed_gpus, build_class, digest, ready_states, scoped_environment, validate_command


class PriorityTests(unittest.TestCase):
    def test_shared_lane_is_a_single_borrow_scope(self):
        self.assertEqual(borrowed_gpus(4), (4, 5))
        with self.assertRaises(ValueError):
            borrowed_gpus(5)
        self.assertFalse(ready_states({'lanes': {'4': {'state': 'MAINTENANCE_PROBE'},
                                                '5': {'state': 'DOJO_RUNNING'}}}, (4, 5)))

    def test_deployed_preflight_return_contract(self):
        # Check the real snapshotted base interface as well as the extension's
        # CLI unpack; five-value mocks would hide the production mismatch.
        root = Path(__file__).parent
        fixture = root / 'fixtures/maintenance_owner_v3.py'
        if not fixture.exists():
            fixture = root / 'snapshots/maintenance_owner_v3.py'
        base_tree = ast.parse(fixture.read_text())
        function = next(n for n in ast.walk(base_tree) if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
        returned = [n.value for n in ast.walk(function) if isinstance(n, ast.Return)]
        self.assertTrue(any(isinstance(value, ast.Tuple) and len(value.elts) == 6 for value in returned))
        import priority_owner
        extension = ast.parse(Path(priority_owner.__file__).read_text())
        checks = [n for n in ast.walk(extension) if isinstance(n, ast.Assign) and
                  isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and
                  n.value.func.attr == 'preflight' and isinstance(n.targets[0], ast.Tuple)]
        self.assertEqual([len(n.targets[0].elts) for n in checks], [6])

    def test_command_rejects_scope_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            command = dict(action='probe', priority_gpus=[4, 5], argv=[str(project / 'python')],
                           run_dir=str(project / 'runs/signal'), cleanup_dir=str(project / 'runs/signal/cleanup'),
                           sweep_id='signal', timeout_s=100)
            validate_command(command, (4, 5), project)
            with self.assertRaises(ValueError):
                validate_command({**command, 'priority_gpus': [4]}, (4, 5), project)
            Path(command['run_dir']).mkdir(parents=True)
            with self.assertRaises(ValueError):
                validate_command(command, (4, 5), project)

    def test_peer_removed_before_signal(self):
        events = []
        class Base:
            def take_over(self):
                self.owner = types.SimpleNamespace(
                    processes={4: types.SimpleNamespace(identity={'pid': 4}),
                               5: types.SimpleNamespace(identity={'pid': 5})},
                    state={'lanes': {'5': {}}}, save=lambda: events.append('saved'))
        def exact_signal(module, identity, signum):
            self.assertNotIn(5, instance.owner.processes)
            events.append(('signal', identity['pid']))
        cls = build_class(types.SimpleNamespace(Maintenance=Base, exact_signal=exact_signal))
        instance = object.__new__(cls)
        instance.gpu, instance.borrowed, instance.peer_processes = 4, (4, 5), {}
        instance.directory, instance.frozen, instance.signal_command = Path('private'), {}, {}
        instance.post_signal_command = None
        instance.M = types.SimpleNamespace(same=lambda ident: True, atomic=lambda *args: None)
        instance.take_over()
        self.assertIn(5, instance.peer_processes)
        self.assertIn(('signal', 5), events)

    def test_resume_checks_both_and_corrects_scope(self):
        events = []
        class Base:
            def resume_formal(self, environment):
                events.append(('resume', self.gpu, environment['DOJO_GPU_SCOPE'], str(self.args.config_path)))
        cls = build_class(types.SimpleNamespace(Maintenance=Base))
        instance = object.__new__(cls)
        instance.gpu, instance.borrowed = 4, (4, 5)
        instance.project, instance.directory = Path('project'), Path('private')
        records = [{'path': f'result-gpu{gpu}.json', 'sha256': 'saved'} for gpu in (4, 5)]
        instance.peer_results = records[1:]
        instance.args = types.SimpleNamespace(config_path=Path('project/scripts/lanes/configs/gpu4.json'))
        instance.frozen_check = lambda: events.append('frozen')
        instance.M = types.SimpleNamespace(gpu_health=lambda g: events.append(('health', g)),
            atomic=lambda *args: None, read=lambda *args: {'results': records}, sha=lambda path: 'saved')
        instance.resume_formal({'DOJO_GPU_SCOPE': '4', 'PYTHONPATH': '/scope'})
        self.assertEqual(events[:3], ['frozen', ('health', 4), ('health', 5)])
        self.assertEqual([(row[1], row[2]) for row in events if isinstance(row, tuple) and row[0] == 'resume'],
                         [(4, '4'), (5, '5')])
        self.assertEqual(instance.gpu, 4)
        self.assertTrue(str(instance.args.config_path).endswith('gpu4.json'))

    def resume_race_instance(self, mutate_gpu):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        results = {gpu: root / f'result-gpu{gpu}.json' for gpu in (4, 5)}
        for gpu, path in results.items():
            path.write_text(json.dumps({'gpu': gpu, 'episodes': 1}))
        records = [{'path': str(path), 'sha256': digest(path)} for path in results.values()]
        proof_path = root / 'pause-proof.json'
        proof_path.write_text(json.dumps({'results': records, 'borrowed_gpus': [4, 5]}))
        proof_bytes = proof_path.read_bytes()
        events = []

        class Base:
            def resume_formal(self, environment):
                checked = self.M.read(self.directory / 'pause-proof.json')['results']
                events.append(('checked', self.gpu, [row['path'] for row in checked]))
                for row in checked:
                    if self.M.sha(Path(row['path'])) != row['sha256']:
                        raise RuntimeError('Saved formal result changed')
                events.append(('resume', self.gpu))
                if self.gpu == 4:
                    results[mutate_gpu].write_text(json.dumps({'gpu': mutate_gpu, 'episodes': 2}))
                self.M.atomic(self.directory / 'formal-resumed.json', {'gpu': self.gpu})

        cls = build_class(types.SimpleNamespace(Maintenance=Base))
        instance = object.__new__(cls)
        instance.gpu, instance.borrowed = 4, (4, 5)
        instance.project = instance.directory = root
        instance.peer_results = records[1:]
        instance.args = types.SimpleNamespace(config_path=root / 'scripts/lanes/configs/gpu4.json')
        instance.frozen_check = lambda: None
        original_read = lambda path: json.loads(Path(path).read_text())
        instance.M = types.SimpleNamespace(read=original_read, sha=digest, gpu_health=lambda gpu: None,
            atomic=lambda path, value: Path(path).write_text(json.dumps(value)))
        instance.owner = types.SimpleNamespace(state={'lanes': {'4': {}, '5': {}}}, save=lambda: None)
        return instance, events, results, proof_bytes, original_read

    def test_resumed_primary_results_do_not_block_paused_peer(self):
        instance, events, results, proof_bytes, original_read = self.resume_race_instance(4)
        instance.resume_formal({'DOJO_GPU_SCOPE': '4'})
        self.assertEqual([event for event in events if event[0] == 'resume'], [('resume', 4), ('resume', 5)])
        self.assertEqual([event for event in events if event[0] == 'checked'],
                         [('checked', 4, [str(results[4])]), ('checked', 5, [str(results[5])])])
        self.assertEqual((instance.directory / 'pause-proof.json').read_bytes(), proof_bytes)
        self.assertIs(instance.M.read, original_read)
        self.assertEqual(instance.gpu, 4)

    def test_paused_peer_result_change_still_rejects_its_resume(self):
        instance, events, results, proof_bytes, original_read = self.resume_race_instance(5)
        with self.assertRaisesRegex(RuntimeError, 'Saved formal result changed'):
            instance.resume_formal({'DOJO_GPU_SCOPE': '4'})
        self.assertEqual([event for event in events if event[0] == 'resume'], [('resume', 4)])
        self.assertEqual(instance.owner.state['lanes']['5']['state'], 'NEEDS_ATTENTION')
        self.assertEqual((instance.directory / 'pause-proof.json').read_bytes(), proof_bytes)
        self.assertIs(instance.M.read, original_read)
        self.assertEqual(instance.gpu, 4)
        self.assertEqual(instance.args.config_path.name, 'gpu4.json')

    def test_all_paused_results_checked_before_first_dispatch(self):
        instance, events, results, proof_bytes, original_read = self.resume_race_instance(4)
        results[5].write_text('changed before either dispatch')
        with self.assertRaisesRegex(RuntimeError, 'changed before priority resume'):
            instance.resume_formal({'DOJO_GPU_SCOPE': '4'})
        self.assertEqual(events, [])
        self.assertTrue(all(row['state'] == 'NEEDS_ATTENTION'
                            for row in instance.owner.state['lanes'].values()))
        self.assertEqual((instance.directory / 'pause-proof.json').read_bytes(), proof_bytes)
        self.assertIs(instance.M.read, original_read)

    def test_post_handoff_exception_retains_attention_owner(self):
        receipts = []
        class StopHeartbeat(BaseException):
            pass
        class Base:
            def run(self):
                self.owner = types.SimpleNamespace(
                    state={'identity': {'pid': 123}, 'lanes': {'4': {}, '5': {}}}, save=lambda: None)
                raise RuntimeError('Peer identity became unreadable after handoff')
        cls = build_class(types.SimpleNamespace(Maintenance=Base))
        instance = object.__new__(cls)
        instance.borrowed, instance.directory = (4, 5), Path('private')
        instance.M = types.SimpleNamespace(atomic=lambda path, data: receipts.append(data))
        with patch('priority_owner.time.sleep', side_effect=StopHeartbeat):
            with self.assertRaises(StopHeartbeat):
                instance.run()
        self.assertEqual(instance.owner.state['stage'], 'NEEDS_ATTENTION')
        self.assertTrue(all(row['state'] == 'NEEDS_ATTENTION' for row in instance.owner.state['lanes'].values()))
        self.assertEqual(receipts[0]['identity']['pid'], 123)

    def probe_instance(self, stage, returncode=0):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run = root / 'run'
        run.mkdir()
        (root / 'probe-example.log').write_text('ordinary clean log')
        cls = build_class(types.SimpleNamespace(Maintenance=type('Base', (), {})))
        instance = object.__new__(cls)
        events = []
        instance.gpu, instance.borrowed = 6, (6,)
        instance.directory, instance.cancel = root, False
        instance.probe_stage, instance.post_started = stage, stage == 'post_signal'
        instance.post_signal_command = {'nonce': 'remaining'}
        instance.probe_cmd = {'nonce': 'example', 'run_dir': str(run)}
        instance.probe = types.SimpleNamespace(poll=lambda: returncode, returncode=returncode)
        instance.cleanup_probe = lambda reason: events.append(('cleanup', reason))
        instance.fatal_pattern = re.compile('CUDA error')
        instance.M = types.SimpleNamespace(gpu_health=lambda gpu: {'gpu': gpu},
            atomic=lambda *args: events.append(('receipt', str(args[0]))),
            read=lambda path: json.loads(Path(path).read_text()))
        instance.owner = types.SimpleNamespace(state={'lanes': {'6': {}}}, save=lambda: None,
            return_lane=lambda *args: events.append(('RLT', args)))
        instance.resume_environment = {'DOJO_GPU_SCOPE': '6'}
        instance.resume_formal = lambda env: events.append(('Dojo', env))
        return instance, events, run

    def test_signal_completion_starts_post_without_dojo_or_rlt(self):
        instance, events, run = self.probe_instance('signal')
        instance.start_post_probe = lambda: events.append(('post', 'started'))
        instance.probe_tick()
        self.assertIn(('post', 'started'), events)
        self.assertFalse(any(event[0] in ('Dojo', 'RLT') for event in events))

    def test_post_completion_requires_receipt_before_dojo(self):
        instance, events, run = self.probe_instance('post_signal')
        (run / 'summary.json').write_text(json.dumps({'state': 'COMPLETE', 'released': True}))
        instance.probe_tick()
        self.assertTrue(any(event[0] == 'Dojo' for event in events))
        self.assertFalse(any(event[0] == 'RLT' for event in events))

    def test_post_failure_does_not_resume_dojo_or_rlt(self):
        instance, events, run = self.probe_instance('post_signal', returncode=2)
        instance.probe_tick()
        self.assertEqual(instance.phase, 'NEEDS_ATTENTION')
        self.assertFalse(any(event[0] in ('Dojo', 'RLT') for event in events))

    def test_post_missing_receipt_does_not_resume_dojo(self):
        instance, events, run = self.probe_instance('post_signal')
        instance.probe_tick()
        self.assertEqual(instance.phase, 'NEEDS_ATTENTION')
        self.assertFalse(any(event[0] in ('Dojo', 'RLT') for event in events))


if __name__ == '__main__':
    unittest.main()
