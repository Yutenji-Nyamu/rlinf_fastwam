import importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
def load(name):
    s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e=load('extend_budget');base=load('rlt_lease')
class BudgetTests(unittest.TestCase):
    def test_absolute_total_and_resume(self):
        original={'runner':{'max_steps':800,'max_epochs':1000,'resume_dir':None},'algorithm':{'alpha_end_round':500,'update_epoch':5}}
        cfg=e.changed_config(original,Path('/x/old'),Path('/x/new'),800,2000)
        self.assertEqual(cfg['runner']['max_steps'],2000);self.assertEqual(cfg['runner']['max_epochs'],2000)
        self.assertTrue(cfg['runner']['resume_dir'].endswith('/old/old/checkpoints/global_step_800'))
        self.assertEqual(original['algorithm'],cfg['algorithm']);self.assertEqual(original['runner']['max_steps'],800)
    def test_completion_failure_and_unreleased(self):
        t={'operation_id':'x','gpu':5,'released':True,'decision':'COMPLETE'}
        self.assertEqual(e.decision(t,{'exit_code':0},'x',5),'continue')
        self.assertEqual(e.decision(t,{'exit_code':1},'x',5),'return')
        self.assertEqual(e.decision({**t,'decision':'FAILED'},{'exit_code':0},'x',5),'return')
        with self.assertRaises(AssertionError):e.decision({**t,'released':False},{'exit_code':0},'x',5)
    def test_lease_fences_original_return(self):
        with tempfile.TemporaryDirectory() as d:
            with base._lock(Path(d)):
                with self.assertRaises(BlockingIOError):base._lock(Path(d))
            with base._lock(Path(d)):pass
    def fixture(self,d):
        root=Path(d);old=root/'old';new=root/'new';stage=root/'stage';stage.mkdir()
        p={'lane':'bc','pins':{},'runs':{'formal':{'run':str(new)}}}
        ext={'original_run':str(old),'from_step':100,'total_steps':400,'retention_authorized':'latest2+resume+final; only this BC; preserve logs'}
        def cp(run,step):
            q=run/run.name/'checkpoints'/f'global_step_{step}'
            for rel in ('actor/model_state_dict/full_weights.pt','actor/local_shard_checkpoint/checkpoint_rank_0.pt',
                        'actor/online_bc/rank_0/success_replay.pt','actor/online_bc/rank_0/learner.pt','actor/online_bc/rank_0/ugrow.pt'):
                f=q/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'test')
            return q
        return root,old,new,stage,p,ext,cp
    def test_retention_keeps_latest_two_and_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,new,stage,p,ext,cp=self.fixture(d);paths=[cp(old,n) for n in (10,20,30,40)]
            rt=SimpleNamespace(scalars=lambda r:{'train/ugrow/round':[{'value':30}]} if r==old else {})
            e.prune_bc(p,ext,rt,None,stage)
            self.assertFalse(paths[0].exists());self.assertTrue(all(q.exists() for q in paths[1:]))
            self.assertTrue(json.loads((stage/'prune-step-10.json').read_text())['deleted'])
    def test_resume_anchor_and_logs_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,new,stage,p,ext,cp=self.fixture(d);a=cp(old,90);b=cp(old,100);c=cp(new,110);f=cp(new,120)
            (old/'metrics.log').write_text('keep')
            rt=SimpleNamespace(scalars=lambda r:{'train/ugrow/round':[{'value':100 if r==old else 120}]})
            e.prune_bc(p,ext,rt,None,stage)
            self.assertFalse(a.exists());self.assertTrue(all(q.exists() for q in (b,c,f,old/'metrics.log')))
    def test_pinned_checkpoint_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,new,stage,p,ext,cp=self.fixture(d);a=cp(old,10);cp(old,20);cp(old,30)
            p['pins'][str(a/'actor/model_state_dict/full_weights.pt')]='immutable'
            rt=SimpleNamespace(scalars=lambda r:{'train/ugrow/round':[{'value':30}]} if r==old else {})
            with self.assertRaises(AssertionError):e.prune_bc(p,ext,rt,None,stage)
            self.assertTrue(a.exists())
if __name__=='__main__':unittest.main()
