"""Exercise complete evaluation bookkeeping on CPU, including frozen configs."""
import dataclasses,importlib.util,json,sys,tempfile,contextlib,io
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
source=Path(__file__).resolve().parents[2];sys.path.insert(0,str(source))
spec=importlib.util.spec_from_file_location('eval_under_test',source/'tools/teacher_eval/evaluate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@dataclasses.dataclass(frozen=True)
class Config:use_dsrl:bool=True
class Model:
 def __init__(self):self.config=Config();self.dsrl_policy_phase=torch.tensor(1)
 def to(self,*a):return self
 def predict_action_batch(self,obs,**kw):return torch.ones(len(obs['states']),20,14)*int(self.config.use_dsrl),{}
class Env:
 def __init__(self,seeds):self.seeds=seeds;self.venv=self;self.success_once=torch.zeros(len(seeds),dtype=torch.bool);self.steps=0
 def _call(self,method):return [dict(accepted_seed=s,instruction='switch',native_action_count=self.steps,success=bool(self.success_once[i]))for i,s in enumerate(self.seeds)]
 def reset(self):return dict(states=torch.zeros(len(self.seeds),14),main_images=torch.zeros(len(self.seeds),2,2,3,dtype=torch.uint8)),{}
 def chunk_step(self,actions):self.steps+=20;self.success_once|=actions[:,0,0].bool();return [self.reset()[0]],None,None,None,None
 def offload(self,**kw):pass
with tempfile.TemporaryDirectory()as td:
 td=Path(td);cfg=OmegaConf.create(dict(env={'eval':{'max_episode_steps':200}},rollout={'model':{'openpi':{'dsrl_eval_deterministic':False}}}));OmegaConf.save(cfg,td/'cfg.yaml')
 job=dict(config=str(td/'cfg.yaml'),output=str(td/'out'),kind='dsrl',checkpoint='unused',chunk_steps=20,seeds=[3,5],repeats=2,parallel=3,noise_seed=42)
 (td/'job.json').write_text(json.dumps(job))
 with patch.object(m,'build_models',return_value=dict(model=Model(),chunk_steps=20,loaded_keys=['test'])),patch.object(m,'environment',side_effect=lambda cfg,seeds:Env(seeds)),patch.object(torch,'set_num_threads'),patch.object(torch,'set_num_interop_threads'),patch.object(torch.random,'fork_rng',side_effect=lambda **kw:contextlib.nullcontext()),patch.object(torch.cuda,'manual_seed_all'),patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(sys,'argv',['eval','--job',str(td/'job.json')]),contextlib.redirect_stdout(io.StringIO()):m.main()
 report=json.loads((td/'out/complete.json').read_text());assert report['base']['episodes']==4 and report['teacher']['successes']==4 and report['difference']==1
 print('Full CPU evaluation lifecycle passed: two conditions, uneven batches, 4 paired records, frozen config, serialization')


