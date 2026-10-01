"""Server-only CPU checks; create only own short-lived subprocesses, no Ray/GPU job."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
from common import ROOT, UID, ProcessIdentityChanged, ProcessIdentityUncertain, account, alive, identity, own_path, pidfd_open, pidfd_send
from wm_stage import Catalog, TAG, run_stage
from prepare_switch import verify_prior_return


class OwnershipChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='wm-resource-cpu-', dir=ROOT)
        self.root = Path(self.tmp.name); self.children = []
        self.token = uuid.uuid4().hex

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
            if child.stdout is not None: child.stdout.close()
            if child.stderr is not None: child.stderr.close()
        self.tmp.cleanup()

    def sleeper(self, token):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(90)'],
                                 env={**os.environ, TAG: token}, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        self.children.append(child)
        return child

    def test_token_and_identity_do_not_signal_unrelated(self):
        owned = self.sleeper(self.token); unrelated = self.sleeper(uuid.uuid4().hex)
        catalog = Catalog(self.root, self.token); catalog.scan()
        rows = {r['pid']: r for r in catalog.rows.values()}
        self.assertIn(owned.pid, rows); self.assertNotIn(unrelated.pid, rows)
        catalog.signal({**rows[owned.pid], 'start': rows[owned.pid]['start'] + 1}, signal.SIGTERM)
        self.assertIsNone(owned.poll(), 'Stale identity must not receive a signal')
        catalog.signal(rows[owned.pid], signal.SIGTERM); owned.wait(timeout=5)
        self.assertIsNone(unrelated.poll(), 'Unrelated process must remain alive')

    def test_permission_unclaimed_skip_registered_identity_reverified(self):
        child = self.sleeper(self.token); blocked = Path('/proc') / str(child.pid) / 'environ'
        original = Path.read_bytes
        def denied(path):
            if path == blocked: raise PermissionError(13, 'CPU test', str(path))
            return original(path)
        catalog = Catalog(self.root, self.token)
        with patch.object(Path, 'read_bytes', denied):
            catalog.scan()
            self.assertNotIn(child.pid, {r['pid'] for r in catalog.rows.values()})
            report = json.loads((self.root / 'managed-identities.json').read_text())
            self.assertIn(child.pid, {r['pid'] for r in report['unreadable_unclaimed']})
            catalog.add(identity(child.pid), 'explicit-test-child')
            catalog.scan()
            report = json.loads((self.root / 'managed-identities.json').read_text())
            self.assertIn(child.pid, {r['pid'] for r in report['unreadable_registered']})
            self.assertIsNone(child.poll())

    def test_unreadable_reused_pid_does_not_inherit_ownership(self):
        child = self.sleeper(self.token)
        catalog = Catalog(self.root, self.token)
        current = identity(child.pid)
        catalog.add({**current, 'start': current['start'] - 1}, 'stale-generation')
        blocked = Path('/proc') / str(child.pid) / 'environ'
        original = Path.read_bytes
        def denied(path):
            if path == blocked: raise PermissionError(13, 'CPU test', str(path))
            return original(path)
        with patch.object(Path, 'read_bytes', denied):
            catalog.scan()
        self.assertNotIn((current['pid'], current['start'], current['boot']), catalog.rows)
        report = json.loads((self.root / 'managed-identities.json').read_text())
        self.assertIn(child.pid, {r['pid'] for r in report['unreadable_unclaimed']})
        self.assertIsNone(child.poll())

    def test_token_family_captures_descendants_without_claiming_ancestors(self):
        parent = self.token; inner = parent + '.' + uuid.uuid4().hex
        ancestor = self.sleeper(parent)
        child = self.sleeper(inner)
        grandchild = self.sleeper(inner + '.' + uuid.uuid4().hex)
        sibling = self.sleeper(parent + '.' + uuid.uuid4().hex)
        collision = self.sleeper(parent + 'extra')
        outer_dir = self.root / 'outer'; outer_dir.mkdir()
        inner_dir = self.root / 'inner'; inner_dir.mkdir()
        outer = Catalog(outer_dir, parent); outer.scan()
        outer_pids = {r['pid'] for r in outer.rows.values()}
        self.assertTrue({ancestor.pid, child.pid, grandchild.pid, sibling.pid} <= outer_pids)
        self.assertNotIn(collision.pid, outer_pids)
        nested = Catalog(inner_dir, inner); nested.scan()
        nested_rows = {r['pid']: r for r in nested.rows.values()}
        self.assertEqual(set(nested_rows), {child.pid, grandchild.pid})
        for row in nested_rows.values(): nested.signal(row, signal.SIGTERM)
        child.wait(timeout=5); grandchild.wait(timeout=5)
        self.assertIsNone(ancestor.poll(), 'Nested cleanup must not stop its ancestor')
        self.assertIsNone(sibling.poll(), 'Nested cleanup must not stop sibling stage')
        self.assertIsNone(collision.poll(), 'Prefix without dot is not family ownership')

    def test_explicit_ctypes_pidfd_compatibility(self):
        child = self.sleeper(self.token)
        before = identity(child.pid)
        with patch.object(os, 'pidfd_open', None, create=True), patch.object(signal, 'pidfd_send_signal', None, create=True):
            fd = pidfd_open(child.pid)
            try:
                self.assertFalse(os.get_inheritable(fd))
                self.assertEqual(identity(child.pid)['start'], before['start'])
                pidfd_send(fd, 0)
                pidfd_send(fd, signal.SIGTERM)
                child.wait(timeout=5)
            finally:
                os.close(fd)

    def test_user_path_spelling_preserved(self):
        self.assertEqual(str(own_path(self.root)), str(self.root.absolute()))

    def test_proc_directory_metadata_is_not_uid_authority(self):
        child = self.sleeper(self.token)
        directory = Path('/proc') / str(child.pid)
        original = Path.stat
        def changed_metadata(path, *args, **kwargs):
            if path == directory:
                raise AssertionError('identity must not use mutable proc directory UID')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'stat', changed_metadata):
            self.assertEqual(identity(child.pid)['uid'], UID)

    def test_foreign_uid_is_not_claimed_or_signaled(self):
        child = self.sleeper(self.token); row = identity(child.pid)
        with patch('common.UID', UID + 1):
            with self.assertRaises(ProcessIdentityChanged): identity(child.pid)
            with self.assertRaises(ProcessIdentityUncertain): alive(row)
            self.assertFalse(alive({**row, 'start':row['start']-1}))
            catalog = Catalog(self.root, self.token); catalog.scan()
            self.assertNotIn(child.pid, {item['pid'] for item in catalog.rows.values()})
            with self.assertRaises(ProcessIdentityUncertain): catalog.signal(row, signal.SIGTERM)
        self.assertIsNone(child.poll())

    def test_inconsistent_uid_snapshot_cannot_prove_registered_exit(self):
        import common
        child = self.sleeper(self.token); row = identity(child.pid)
        original = common._proc_contents; status_reads = 0
        def changing(fd, filename):
            nonlocal status_reads
            data = original(fd, filename)
            if filename == 'status':
                status_reads += 1
                if status_reads == 2:
                    data = b'\n'.join(b'Uid:\t%d\t%d\t%d\t%d' % (UID, UID+1, UID, UID)
                        if line.startswith(b'Uid:') else line for line in data.splitlines())
            return data
        with patch('common._proc_contents', changing):
            with self.assertRaises(ProcessIdentityUncertain): alive(row)
        self.assertIsNone(child.poll())

    def test_non_utf8_process_name_is_safe(self):
        code = "import ctypes,time;ctypes.CDLL(None).prctl(15,b'\\xffwm',0,0,0);print('ready',flush=True);time.sleep(90)"
        child = subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,
                                 env={**os.environ,TAG:self.token},start_new_session=True)
        self.children.append(child); self.assertEqual(child.stdout.readline(),b'ready\n')
        self.assertEqual(identity(child.pid)['uid'],UID)
        catalog=Catalog(self.root,self.token);catalog.scan()
        self.assertIn(child.pid,{row['pid'] for row in catalog.rows.values()})

    def test_signal_recheck_preserves_registered_generation(self):
        child = self.sleeper(self.token); row = identity(child.pid)
        catalog = Catalog(self.root, self.token)
        with patch('wm_stage.alive', return_value=True), \
             patch('wm_stage.identity', return_value={**row, 'start': row['start'] + 1}), \
             patch('wm_stage.pidfd_open') as opened:
            self.assertFalse(catalog.signal(row, signal.SIGTERM))
            opened.assert_not_called()
        self.assertIsNone(child.poll())

    def test_cleanup_passes_original_identity_to_signal(self):
        # Inspect the actual call site; no GPU or cleanup action is run here.
        import ast, cleanup_owned, inspect
        tree = ast.parse(inspect.getsource(cleanup_owned))
        signals = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'signal']
        self.assertEqual(len(signals), 1)
        self.assertIsInstance(signals[0].args[0], ast.Name)
        self.assertEqual(signals[0].args[0].id, 'row')

    def test_monitor_and_final_scan_errors_preserved_cleanup_still_attempted(self):
        attempt = self.root / 'attempt'; attempt.mkdir()
        spec = self.stage_spec(cleanup_exit=9)
        # Track only the own CPU children started by this test for final teardown.
        original_popen = subprocess.Popen
        def start(*args, **kwargs):
            child = original_popen(*args, **kwargs); self.children.append(child); return child
        actual_scan = Catalog.scan; calls = 0
        def injected(catalog):
            nonlocal calls
            calls += 1
            if calls == 1: raise RuntimeError('injected monitor observation failure')
            if calls == 2: raise RuntimeError('injected final scan failure')
            return actual_scan(catalog)
        with patch('wm_stage.subprocess.Popen', side_effect=start), patch.object(Catalog, 'scan', injected):
            with self.assertRaisesRegex(AssertionError, 'cleanup callback failed'):
                run_stage(spec, self.root / 'cycle-test', attempt,
                          lambda _: [], lambda *args, **kwargs: None, lambda: False)
        report = json.loads((Path(spec['run_dir']) / 'wm-exit.json').read_text())
        self.assertIn('injected monitor', report['monitor_error']['traceback'])
        self.assertIn('injected final', report['final_scan_error']['traceback'])
        self.assertTrue((Path(spec['run_dir']) / 'cleanup.log').exists())
        self.assertFalse((attempt / 'wm-release.json').exists())

    def test_exact_external_return_and_optional_first_round(self):
        import hashlib
        attempt = self.root / 'prior-attempt'; attempt.mkdir()
        cycle = self.root / 'prior-cycle'; cycle.mkdir()
        def save(path, value): path.write_text(json.dumps(value))
        save(attempt / 'pipeline-final.json', dict(rlt_dispatched=False, error=None, wm_released=True))
        save(attempt / 'user-rlt-return.json', dict(phase='USER_RESTORED_RLT', exit_code=0,
             dispatched=True, cycle=str(cycle), continuation=str(attempt)))
        release = attempt / 'dojo-release.json'
        save(release, dict(all_workers_stopped=True, cleanup_receipt=dict(gpus_released=True)))
        save(cycle / 'resumed-dispatched.json', dict(cycle_id=cycle.name,
             release=dict(path=str(release), sha256=hashlib.sha256(release.read_bytes()).hexdigest())))
        self.assertEqual(verify_prior_return(attempt, cycle, False), 'user-rlt-return.json')
        with self.assertRaises(FileNotFoundError): verify_prior_return(attempt, cycle, True)
        returned = json.loads((attempt / 'user-rlt-return.json').read_text())
        returned['cycle'] = str(self.root / 'other-cycle')
        save(attempt / 'user-rlt-return.json', returned)
        with self.assertRaises(AssertionError): verify_prior_return(attempt, cycle, False)

    def stage_spec(self, cleanup_exit=0):
        callback = self.root / 'callback.py'
        callback.write_text('''import json,os,time
from pathlib import Path
if EXIT: raise SystemExit(EXIT)
receipt={"time":time.time(),"cycle_id":os.environ["WAN_GOAL_CYCLE_ID"],
"owner_token":os.environ["WM_OWNER_TOKEN"],"physical_gpus":[4,5,6,7],
"processes_clear":True,"gpus_released":True,
"ray":{"address":os.environ["RAY_ADDRESS"],"namespace":os.environ["CLUSTER_NAMESPACE"],
"temp_dir":os.environ["WAN_GOAL_RAY_TMPDIR"],"stopped":True}}
Path(os.environ["WAN_GOAL_CLEANUP_RECEIPT"]).write_text(json.dumps(receipt))
'''.replace('EXIT', str(cleanup_exit)))
        return dict(schema=1, run_dir=str(self.root / 'wm-run'), cwd=str(self.root),
                    physical_gpus=[4, 5, 6, 7], max_seconds=None, cleanup_timeout_seconds=15,
                    command=[sys.executable, '-c', 'import time;time.sleep(1.1);raise SystemExit(17)'],
                    cleanup_command=[sys.executable, '-u', str(callback)], environment={},
                    ray=dict(isolation='dedicated', address='127.0.0.1:61997',
                             namespace='cpu-test-' + self.token, temp_dir=str(self.root / 'ray-tmp')))

    def test_failed_wm_with_verified_cleanup_returns_for_dojo(self):
        attempt = self.root / 'attempt'; attempt.mkdir(); states = []
        result = run_stage(self.stage_spec(), self.root / 'cycle-test', attempt,
                           lambda gpus: [], lambda phase, **kw: states.append(phase), lambda: False)
        self.assertEqual(result['wm_exit_code'], 17)
        self.assertEqual(result['outcome'], 'failed')
        self.assertEqual(result['next'], 'RESUME_EXISTING_DOJO')
        self.assertTrue((attempt / 'wm-release.json').exists())

    def test_cleanup_failure_never_produces_release(self):
        attempt = self.root / 'attempt'; attempt.mkdir()
        with self.assertRaisesRegex(AssertionError, 'cleanup callback failed'):
            run_stage(self.stage_spec(9), self.root / 'cycle-test', attempt,
                      lambda gpus: [], lambda phase, **kw: None, lambda: False)
        self.assertFalse((attempt / 'wm-release.json').exists())


if __name__ == '__main__':
    account()
    unittest.main(verbosity=2)
