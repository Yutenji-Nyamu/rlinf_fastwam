"""CPU-only identity/scope fixtures. Run with the SZ3 RLT Python; no Ray/GPU import."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('rlt567', Path(__file__).with_name('rlt_gpu567_cycle.py'))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stage = self.root / 'cycle-fixture'
        self.stage.mkdir()
        self.helper = types.SimpleNamespace(same=Mock(return_value=False), gpu_processes=Mock(return_value=[]),
                                            now=lambda: 'fixture')
        self.addCleanup(patch.stopall)
        patch.object(C, 'ROOT', self.root).start()
        patch.object(C, 'H', self.helper).start()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream)

    def rows(self):
        return {f'gpu{gpu}': {'gpus': [gpu], 'original_run': f'/old/{gpu}',
                'original_namespace': f'old-{gpu}', 'new_run': f'/new/{gpu}', 'namespace': f'new-{gpu}'}
                for gpu in C.GPUS}

    def release(self, processes, terminal='completed'):
        path = self.root / 'release.json'
        self.save(path, {'cycle_id': self.stage.name, 'gpus': [5, 6, 7], 'terminal_status': terminal,
                         'all_workers_stopped': True, 'managed_processes': processes})
        return path

    def test_owner_requires_pid_uid_start_and_bound_invocation(self):
        expected = dict(C.EXPECTED_OWNER, cmdline_sha256='a', match_cmdline=True)
        current = dict(expected, state='S')
        self.assertEqual(C.validate_owner_identity(expected, current), current)
        for field, wrong in [('pid', 999), ('uid', 0), ('start', 0), ('cmdline_sha256', 'different'), ('state', 'Z')]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                C.validate_owner_identity(expected, dict(current, **{field: wrong}))
        with self.assertRaises(AssertionError):
            C.validate_owner_identity(dict(expected, start=0), current)

    def test_retire_uses_only_exact_owner_pidfd(self):
        identity = dict(C.EXPECTED_OWNER, state='S', cmdline_sha256='a', match_cmdline=True)
        self.helper.proc = Mock(return_value=identity)
        self.helper.same = Mock(return_value=False)
        self.helper.save = self.save
        plan = {'old_owner': identity, 'runs': {k: dict(r, original_identity={'pid': 100 + i})
                                               for i, (k, r) in enumerate(self.rows().items())}}
        with patch.object(C, 'pidfd_open', return_value=12) as opened, \
             patch.object(C, 'pidfd_send') as sent, patch.object(C.os, 'close') as closed, \
             patch.object(C.os, 'kill') as raw_kill:
            C.retire_monitor(self.stage, plan)
        opened.assert_called_once_with(3476669)
        sent.assert_called_once_with(12, signal.SIGTERM)
        closed.assert_called_once_with(12)
        raw_kill.assert_not_called()
        self.assertTrue(C.read(self.stage / 'monitor-retired.json')['monitor_only'])

    def test_retire_rechecks_identity_after_pidfd_open(self):
        identity = dict(C.EXPECTED_OWNER, state='S')
        self.helper.proc = Mock(side_effect=[identity, dict(identity, start=identity['start'] + 1)])
        self.helper.save = self.save
        with patch.object(C, 'pidfd_open', return_value=12), \
             patch.object(C, 'pidfd_send') as sent, patch.object(C.os, 'close'):
            with self.assertRaises(AssertionError):
                C.retire_monitor(self.stage, {'old_owner': identity, 'runs': {}})
        sent.assert_not_called()
        self.assertFalse((self.stage / 'monitor-retire-attempt.json').exists())

    def test_pidfd_normal_and_forced_fallback_signal_only_fresh_child(self):
        def proc(pid):
            try:
                path = Path('/proc') / str(pid)
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                return {'pid': pid, 'uid': path.stat().st_uid, 'start': int(fields[19]), 'state': fields[0]}
            except (FileNotFoundError, ProcessLookupError):
                return None

        def same(expected):
            current = proc(expected['pid'])
            return bool(current and current['state'] not in ('Z', 'X') and
                        all(current[k] == expected[k] for k in ('pid', 'uid', 'start')))

        self.helper.proc, self.helper.same, self.helper.save = proc, same, self.save
        for forced in (False, True):
            with self.subTest(forced_fallback=forced):
                child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                identity = proc(child.pid)
                self.assertEqual(identity['uid'], os.getuid())
                phase = self.stage / ('forced' if forced else 'available')
                phase.mkdir()
                patches = []
                if forced:
                    patches = [patch.object(C.os, 'pidfd_open', None, create=True),
                               patch.object(C.signal, 'pidfd_send_signal', None, create=True)]
                for item in patches:
                    item.start()
                try:
                    C.pidfd_probe()
                    fd = C.pidfd_open(child.pid)
                    try:
                        self.assertFalse(os.get_inheritable(fd))
                        C.pidfd_send(fd, 0)
                    finally:
                        os.close(fd)
                    with patch.object(C, 'EXPECTED_OWNER', {k: identity[k] for k in ('pid', 'uid', 'start')}):
                        for field in ('start', 'uid'):
                            with self.assertRaises(AssertionError):
                                C.retire_monitor(phase, {'old_owner': dict(identity, **{field: identity[field] + 1}), 'runs': {}})
                            self.assertIsNone(child.poll(), 'Wrong identity must remain untouched')
                        C.retire_monitor(phase, {'old_owner': identity, 'runs': {}})
                    self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)
                finally:
                    if same(identity):
                        fd = C.pidfd_open(child.pid)
                        try:
                            if same(identity):
                                C.pidfd_send(fd, signal.SIGKILL)
                        finally:
                            os.close(fd)
                    child.wait(timeout=5)
                    for item in reversed(patches):
                        item.stop()

    def test_release_rejects_live_managed_process(self):
        self.helper.same.return_value = True
        path = self.release([{'pid': 1, 'uid': 20001, 'start': 1}])
        with self.assertRaisesRegex(AssertionError, 'process remains'):
            C.validate_release(self.stage, path)

    def test_release_checks_567_compute_and_graphics_without_gpu4(self):
        path = self.release([{'pid': 1, 'uid': 20001, 'start': 1}])
        self.helper.gpu_processes = Mock(side_effect=lambda gpus: [
            {'gpu': 4, 'pid': 90, 'type': 'C+G'}] if 4 in gpus else [])
        self.assertEqual(C.validate_release(self.stage, path)['terminal_status'], 'completed')
        self.helper.gpu_processes.assert_called_once_with([5, 6, 7])
        self.helper.gpu_processes = Mock(return_value=[{'gpu': 6, 'pid': 2, 'type': 'G'}])
        with self.assertRaisesRegex(AssertionError, 'C/G context'):
            C.validate_release(self.stage, path)

    def test_three_watch_routes_exact_and_gpu4_untouched(self):
        rows = self.rows()
        watch = {'runs': {k: {'gpus': r['gpus'], 'run': r['original_run'], 'namespace': r['original_namespace']}
                          for k, r in rows.items()}}
        watch['runs']['gpu4'] = {'gpus': [4], 'run': '/unrelated/4', 'namespace': 'unrelated'}
        before = copy.deepcopy(watch)
        self.assertEqual(set(C.watch_targets(watch, rows)), set(rows))
        self.assertEqual(watch, before)
        watch['runs']['gpu6'].update(run='/new/6', namespace='new-6')
        C.watch_targets(watch, rows, allow_returned=True)
        with self.assertRaises(AssertionError):
            C.watch_targets(watch, rows)
        watch['runs']['gpu6']['namespace'] = 'wrong'
        with self.assertRaises(AssertionError):
            C.watch_targets(watch, rows, allow_returned=True)

    def test_stop_binds_all_three_runs_and_retires_monitor_first(self):
        rows = self.rows()
        for i, row in enumerate(rows.values()):
            row.update(original_identity={'pid': 100 + i, 'uid': 20001, 'start': i + 1}, original_jobs=[f'job{i}'])
            pre = self.stage / 'prepared' / f'gpu{row["gpus"][0]}'
            pre.mkdir(parents=True)
            (pre / 'original.yaml').write_text('fixture')
            row['prepared_sha256'] = {'original.yaml': C.sha(pre / 'original.yaml')}
        plan = {'runs': rows, 'old_owner': C.EXPECTED_OWNER, 'next_six_ops': 'fake', 'repo': '/fake'}
        self.helper.same = lambda ident: ident['pid'] != C.EXPECTED_OWNER['pid']
        self.helper.actors = lambda _: []
        self.helper.active = lambda *_: []
        self.helper.validate_actor_rows = lambda *_: None
        self.helper.process_tree = lambda roots: {pid: {'pid': pid} for pid in roots}
        self.helper.config = lambda _: {}
        order = []
        ops = types.SimpleNamespace()

        def fake_stop(arg):
            self.assertIsNone(arg)
            order.append('stop')
            bound = ops.checked()['old_runs']
            self.assertEqual([r['gpus'] for r in bound], [[5], [6], [7]])
            self.assertEqual({r['namespace'] for r in bound}, {'old-5', 'old-6', 'old-7'})
            self.assertEqual(ops.gpu_pids([5, 6, 7]), [])

        ops.stop_old = fake_stop
        with patch.object(C, 'load_plan', return_value=plan), patch.object(C, 'read', return_value={}), \
             patch.object(C, 'watch_targets'), patch.object(C, 'require_checkpoint'), \
             patch.object(C, 'retire_monitor', side_effect=lambda *_: order.append('retire')), \
             patch.object(C, 'import_file', return_value=ops), \
             patch.object(C, 'finalize_stopped', return_value={'stopped': True}):
            self.assertTrue(C.stop(self.stage)['stopped'])
        self.assertEqual(order, ['retire', 'stop'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
