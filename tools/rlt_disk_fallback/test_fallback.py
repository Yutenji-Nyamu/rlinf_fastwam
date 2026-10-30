"""CPU-only tests: no checkpoint at end, full-disk transition, priority fencing."""
import errno
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import common
import nosave
import owner


class FallbackTest(unittest.TestCase):
    def test_full_disk_control_receipt(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {'RLT_EMERGENCY_ROOT': td+'/root'}):
            target=Path(td)/'data'/'status.json'
            original=Path.write_text
            def write(p,*args,**kwargs):
                if 'data' in p.parts:raise OSError(errno.ENOSPC,'test full data')
                return original(p,*args,**kwargs)
            with patch.object(Path,'write_text',write):common.save(target,{'phase':'RLT_RUNNING'})
            self.assertEqual(common.read(target),{'phase':'RLT_RUNNING'})

    def test_real_configs_and_native_save_gate(self):
        from rlinf.utils.runner_utils import check_progress
        from omegaconf import OmegaConf
        live=common.read('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json')
        for g,slot in live['slots'].items():
            q=common.read(slot['fallback']);cfg=common.read(Path(q['runtime'])/'resolved.yaml')
            cp=nosave.latest_complete(q)
            actual=nosave.make_config(cfg,Path('/tmp/test-nosave')/g,cp)
            OmegaConf.to_container(OmegaConf.create(actual),resolve=True)
            self.assertEqual(cfg['actor'],actual['actor'])
            self.assertEqual(cfg['rollout'],actual['rollout'])
            self.assertFalse(actual['algorithm']['replay_buffer']['auto_save'])
            self.assertFalse(actual['algorithm']['dv_observe'])
            self.assertEqual(actual['runner']['resume_dir'],str(cp))
            for step in (25,3000,actual['runner']['max_epochs']):
                self.assertFalse(check_progress(step,actual['runner']['max_epochs'],25,-1,1.0)[1])
            print(f'GPU{g}: CP{cp.name.split("_")[-1]}, training config retained, periodic/final saves disabled')

    def run_owner_once(self,initial,external=False,full=False):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);request=root/'fallback.json';common.save(request,{'runtime':td,'namespace':'test','kind':'rlt'})
            slot={'fallback':str(request),'gpu_uuid':'GPU-test','no_checkpoint_fallback':True}
            if external:slot.update(external_release=str(root/'released.json'),external_identity={'pid':123})
            plan={'uid':os.getuid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'slots':{'4':slot},'poll_seconds':2}
            common.save(root/'plan.json',plan);common.save(root/'status.json',{'slots':{'4':initial}});common.save(root/'fallback-enabled.json',{})
            with patch.object(sys,'argv',['owner','--plan',str(root/'plan.json')]),patch.object(owner,'gpus',return_value={4:{'uuid':'GPU-test','processes':[]}}),patch.object(owner,'same',return_value=False),patch.object(owner,'actors',return_value=[]),patch.object(owner,'releasable',return_value=True),patch.object(owner,'prepare_nosave',return_value='nosave-request') as prepared,patch.object(owner,'launch',return_value=({'pid':999},None)) as launched,patch.object(owner.time,'sleep',side_effect=KeyboardInterrupt):
                if full:launched.side_effect=[OSError(errno.ENOSPC,'full data'),({'pid':999},None)]
                try:owner.main()
                except KeyboardInterrupt:pass
                return common.read(root/'status.json'),prepared.call_count,launched.call_args_list

    def test_tail_immediately_follows_rlt_exit(self):
        # request needs to survive the temporary owner fixture; use a separate file.
        with tempfile.TemporaryDirectory() as td:
            request=Path(td)/'old.json';common.save(request,{'namespace':'old'})
            state,prepared,launched=self.run_owner_once({'phase':'RLT_RUNNING','identity':{'pid':123},'request':str(request)})
        self.assertEqual(prepared,1);self.assertEqual(state['slots']['4']['phase'],'RLT_RUNNING')
        self.assertTrue(state['slots']['4']['fallback_no_save'])

    def test_priority_is_not_preempted(self):
        state,prepared,launched=self.run_owner_once({'phase':'WAIT_EXTERNAL'},external=True)
        self.assertEqual(prepared,0);self.assertFalse(launched)

    def test_full_disk_at_normal_launch(self):
        state,prepared,launched=self.run_owner_once({'phase':'WAITING_RLT'},full=True)
        self.assertEqual(prepared,1);self.assertEqual(len(launched),2)
        self.assertEqual(state['slots']['4']['request'],'nosave-request')

    def test_independent_bounded_log_sink(self):
        with tempfile.TemporaryDirectory() as td:
            producer=subprocess.Popen([sys.executable,'-c','import sys; sys.stdout.buffer.write(b"x"*18000000)'],stdout=subprocess.PIPE)
            sink=nosave.drain_log(producer.stdout,Path(td)/'driver.log')
            self.assertEqual(producer.wait(timeout=15),0)
            self.assertEqual(sink.wait(timeout=15),0)
            self.assertLessEqual(sum(p.stat().st_size for p in Path(td).glob('driver.log*')),16*1024**2)


if __name__=='__main__':unittest.main()
