"""CPU regressions for timeout isolation and identity-exact termination."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('lean_owner',Path(__file__).with_name('owner.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def proc(pid):
    try:
        p=Path('/proc')/str(pid); s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,uid=p.stat().st_uid,start=int(s[19]),state=s[0],ppid=int(s[1]))
    except (FileNotFoundError,ProcessLookupError):
        return None

def same(i):
    p=proc(i['pid'])
    return bool(p and p['uid']==i['uid'] and p['start']==i['start'] and p['state'] not in ('Z','X'))

class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        M.H=types.SimpleNamespace(proc=proc,same=same,now=lambda:'test',atomic=lambda *a:None)
        self.plan={'owner_dir':self.temp.name,'trials':[{'timeout_seconds':100}]}

    def tearDown(self):
        self.temp.cleanup()

    def test_dashboard_and_resource_timeout_do_not_interrupt_training(self):
        M.H.actors=Mock(side_effect=AssertionError('Dashboard must never be queried'))
        M.H.gpu_processes=Mock(side_effect=TimeoutError('nvidia-smi timed out'))
        cat=Mock(); cat.live.return_value=[]
        driver=Mock(returncode=0);driver.poll.side_effect=[None,None,0]
        wm=Mock();wm.poll.return_value=None
        with patch.object(M.time,'sleep'):
            self.assertEqual(M.wait_training(self.plan,driver,wm,cat),0)
        M.H.actors.assert_not_called()

    def test_actual_service_exit_is_fatal(self):
        driver=Mock();driver.poll.return_value=None
        wm=Mock();wm.poll.return_value=1
        with self.assertRaisesRegex(RuntimeError,'WM service exited'):
            M.wait_training(self.plan,driver,wm,Mock())

    def test_actual_training_failure_is_preserved(self):
        driver=Mock(returncode=7);driver.poll.return_value=7
        self.assertEqual(M.wait_training(self.plan,driver,Mock(),Mock()),7)

    def test_wrong_gpu_use_is_not_swallowed(self):
        cat=Mock();cat.live.return_value=[dict(pid=9,phase='formal')]
        M.H.gpu_processes=lambda _: [dict(pid=9,gpu=0)]
        with self.assertRaisesRegex(AssertionError,'unassigned GPU'):
            M.observe(self.plan,cat,'formal')

    def test_cleanup_finds_token_worker_and_leaves_foreign_process(self):
        token='cpu-owner-test-'+str(os.getpid())
        env=dict(os.environ,**{M.TOKEN:token,M.PHASE:'formal'})
        command=[sys.executable,'-c','import time; time.sleep(60)']
        own=subprocess.Popen(command,env=env)
        foreign=subprocess.Popen(command)
        try:
            cat=M.Catalog(Path(self.temp.name)/'catalog.json',token)
            cat.scan()
            self.assertIn(own.pid,{r['pid'] for r in cat.live()})
            self.assertNotIn(foreign.pid,{r['pid'] for r in cat.live()})
            self.assertTrue(M.cleanup(self.plan,cat)['all_stopped'])
            own.wait(timeout=3)
            self.assertIsNone(foreign.poll())
        finally:
            for p in (own,foreign):
                if p.poll() is None:p.terminate()
                p.wait(timeout=3)

    def test_reused_pid_is_never_signalled(self):
        cat=M.Catalog(Path(self.temp.name)/'catalog.json','x')
        stale=dict(proc(os.getpid()),start=0)
        with patch.object(M,'pidfd_open') as opened:
            cat.send(stale,15)
        opened.assert_not_called()

if __name__=='__main__':
    unittest.main()
