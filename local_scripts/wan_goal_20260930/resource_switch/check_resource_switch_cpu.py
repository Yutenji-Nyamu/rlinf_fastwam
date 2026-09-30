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
from common import ROOT, account, identity, own_path, pidfd_open, pidfd_send
from wm_stage import Catalog, TAG, run_stage


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

    def test_permission_unclaimed_skip_registered_fail(self):
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
            with self.assertRaisesRegex(RuntimeError, 'registered WM PID ' + str(child.pid)):
                catalog.scan()

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
