import json,os,signal,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
import pair_gate as gate_module
import primary_retry_policy as policy

def put(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj))

class PrimaryRetry(unittest.TestCase):
    def test_no_retry_limit(self):
        self.assertIsNone(policy.terminal_reason({"result":{"phase":"FAILED_RETURN_RLT","error":"Child exited 255"}}))

    def test_explicit_stop_and_normal_completion(self):
        for phase in ('COMPLETE','COMPLETE_RETURN_RLT','TARGET_REACHED'):
            self.assertEqual(policy.terminal_reason({"phase":phase}),'TARGET_REACHED')
        self.assertEqual(policy.terminal_reason({'result':{'phase':'FAILED_RETURN_RLT','error':'Traceback\nSystemExit: 143\n'}}),'OPERATOR_STOP')

    def test_repeated_exits_then_completion(self):
        self.exercise(['unknown','cuda','unknown','cuda','unknown','complete'],5,'TARGET_REACHED')

    def test_handover_and_manual_stop(self):
        self.exercise(['unknown','stop'],1,'OPERATOR_STOP',follow=True)

    def exercise(self,events,count,terminal,follow=False):
        ops=os.environ.get("QUEUE_OPS", "")
        sys.path.insert(0,ops)
        from common import proc
        from contextlib import ExitStack
        kind=os.environ.get('PRIMARY_TEST_KIND','dojo')
        slots=[4,7] if kind=='fastwam' else [5,6]
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            pair=Path(t)/'pair';gate=Path(t)/'gate';gate.mkdir();pair.mkdir()
            cp=pair/'run'/'checkpoints'/'global_step_2'
            put(cp/'dojo-complete.json',{'step':2,'time':1})
            required=['actor/dcp_checkpoint/.metadata','actor/dcp_checkpoint/__0_0.distcp','actor/dcp_checkpoint/__1_0.distcp'] if kind=='fastwam' else ['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/local_shard_checkpoint/checkpoint_rank_1.pt']
            for f in required:
                p=cp/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'test')
            put(pair/'config/formal.yaml',dict(runner=dict(resume_dir=str(cp),max_steps=20,logger=dict(log_path=str(pair),experiment_name='run'))))
            put(pair/'plan.json',dict(gpus=slots,training_source=str(pair),python=sys.executable))
            first=dict(pid=970001,start=100,uid=os.getuid());put(pair/'identity.json',first);put(pair/'registered.json',{'identity':first})
            put(gate/'registered.json',{'identity':proc(os.getpid())})
            option={'mode':'retry_original'}
            if follow:
                old=Path(t)/'previous'/'released.json';put(old,{'phase':'RETRY_FINISHED'})
                option['previous_gate']={'identity':{'pid':970000,'start':60,'uid':os.getuid()},'release':str(old)}
            put(gate/'primary-policy.json',option)
            original_same_calls=0
            def same(w):
                nonlocal original_same_calls
                if w['pid']==first['pid']:
                    original_same_calls+=1;return original_same_calls==1
                return w['pid']!=970000
            state={'index':-1,'launches':0}
            def result(*args):
                state['index']+=1
                event=events[state['index']]
                phase='COMPLETE' if event=='complete' else 'FAILED_RETURN_RLT'
                error='SystemExit: 143' if event=='stop' else 'Child exited 255'
                r={'result':{'phase':phase,'error':error}};put(pair/'released.json',r);return r
            def diagnostic(*args):
                return {'kind':'cuda','line':'CUDA out of memory'} if events[state['index']]=='cuda' else None
            class Child:
                def __init__(self,*args,**kwargs):
                    state['launches']+=1;self.pid=970001+state['launches'];self.code=None
                    put(pair/'identity.json',dict(pid=self.pid,start=100+self.pid,uid=os.getuid()))
                def wait(self,**kwargs):self.code=1;return self.code
                def poll(self):return self.code
            old_handler=signal.getsignal(signal.SIGTERM);stack.callback(signal.signal,signal.SIGTERM,old_handler)
            for target,fn in [('common.same',same),('pair_gate.original_result',result),('pair_gate.oom_evidence',diagnostic),('pair_gate.live_driver',lambda *a:False),('pair_gate.retry_delay',lambda *a:(None,{'external':True})),('pair_gate.replace_active_identity',lambda *a:slots),('pair_gate.subprocess.Popen',Child),('pair_gate.time.sleep',lambda *a:None)]:
                stack.enter_context(mock.patch(target,fn))
            gate_module.run_locked(gate,pair,kind)
            self.assertEqual(state['launches'],count)
            self.assertEqual(json.loads((gate/'released.json').read_text())['phase'],terminal)
            for k in range(1,count+1):
                receipt=json.loads((gate/f'attempt-{k}'/'reason.json').read_text())
                self.assertEqual(receipt['checkpoint'],str(cp))
                self.assertEqual(receipt['delay_seconds'],0)

if __name__=='__main__':unittest.main()
