"""Real CPU data transforms + resolved budget/seed checks; no model inference."""
import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import ray
import torch
import openpi.transforms as transforms
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from openpi.training import checkpoints

from rlinf.config import validate_cfg
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv, partition_success_seeds
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0ForRLActionPrediction

root=Path.cwd()
packet=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-pi05-20260905')
with initialize_config_dir(version_base='1.1',config_dir=str(root/'examples/embodiment/config')):
    cfg=compose(config_name='robotwin_adjust_bottle_online_bc_openpi',overrides=['+online_bc_model=pi05_sidney'])
try: cfg=validate_cfg(cfg)
finally: ray.shutdown()
resolved=OmegaConf.to_container(cfg,resolve=True)
old_run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8')
old=OmegaConf.to_container(OmegaConf.load(old_run/'runtime/resolved.yaml'),resolve=True)
def flatten(x,p=''):
    if isinstance(x,dict): return {a:b for k,v in x.items() for a,b in flatten(v,p+'.'+k if p else k).items()}
    return {p:x}
a,b=flatten(old),flatten(resolved)
delta={k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
allowed={'runner.logger.experiment_name','actor.model.model_path','rollout.model.model_path',
    'actor.model.num_steps','actor.model.openpi.num_steps','actor.model.openpi.config_name',
    'env.train.task_config.task_name','env.eval.task_config.task_name','env.eval.seeds_path'}
for k,(x,y) in delta.items():
    if k in allowed:continue
    assert isinstance(x,str) and isinstance(y,str) and y.replace(str(root),str(root.parent/'pi0-online-bc')).replace(os.environ['ONLINE_BC_RUN_DIR'],str(old_run))==x,(k,x,y)

def seed_env(eval_cfg,n,rank=0,world=1,batches=1):
    env=RoboTwinEnv.__new__(RoboTwinEnv)
    env.cfg=OmegaConf.create(OmegaConf.to_container(eval_cfg,resolve=True))
    env.cfg.fixed_reset_batch_count=batches;env.cfg.rollout_epoch=batches
    env.task_name='move_pillbottle_pad';env.base_seed=eval_cfg.seed
    env.seed=eval_cfg.seed+rank;env.seed_offset=rank
    env.total_num_processes=world;env.group_size=1;env.num_envs=env.num_group=n
    env.auto_reset=False;env.use_fixed_reset_state_ids=True
    env._init_reset_state_ids()
    values=[]
    for _ in range(batches):values+=env.reset_state_ids.tolist();env.update_reset_state_ids()
    assert env.reset_state_ids.tolist()==values[:n]
    return values
sidney_run=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
sidney=OmegaConf.load(sidney_run/'runtime-resume100-to200/resolved.yaml')
sidney_fixed=sum([seed_env(sidney.env.eval,16,rank=i,world=2) for i in range(2)],[])
# Invert the existing deterministic partition permutation, keeping environment
# code untouched and preserving the old two-rank ordered fixed32 on one rank.
permutation=partition_success_seeds(torch.arange(32),base_seed=cfg.env.eval.seed,
    seed_offset=0,total_num_processes=1,num_group=8).tolist()
assert sorted(permutation)==list(range(32))
seed_input=[0]*32
for index,seed in zip(permutation,sidney_fixed):seed_input[index]=seed
seed_path=Path(cfg.env.eval.seeds_path)
assert seed_path==root/'rlinf/envs/robotwin/seeds/eval_sidney_fixed32.json'
seed_text=json.dumps({'move_pillbottle_pad':{'success_seeds':seed_input}},indent=2)+'\n'
if seed_path.exists():assert seed_path.read_text()==seed_text
else:seed_path.write_text(seed_text)
fixed=seed_env(cfg.env.eval,8,batches=4)
assert len(fixed)==len(set(fixed))==32
assert fixed==sidney_fixed, ('Fixed32 differs from Sidney',fixed,sidney_fixed)

native=get_openpi_config('pi05_sidney_robotwin',model_path=cfg.actor.model.model_path)
data=native.data.create(native.assets_dirs,native.model)
assert not data.use_quantile_norm
norm=checkpoints.load_norm_stats(cfg.actor.model.model_path,data.asset_id)
# Use the real data wrappers without allocating VLM/expert weights or any GPU.
model=OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
torch.nn.Module.__init__(model)
model.register_parameter('probe',torch.nn.Parameter(torch.zeros(1)))
model.config=SimpleNamespace(use_rlt=False,action_chunk=50,action_env_dim=14)
model.setup_wrappers(
    transforms=[*data.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=False),*data.model_transforms.inputs],
    output_transforms=[*data.model_transforms.outputs,transforms.Unnormalize(norm,use_quantiles=False),*data.data_transforms.outputs])
commands=torch.linspace(-0.5,0.5,2*50*14).reshape(2,50,14)
raw={'observation/image':torch.zeros(2,224,224,3,dtype=torch.uint8),
     'observation/wrist_image':torch.zeros(2,2,224,224,3,dtype=torch.uint8),
     'observation/state':torch.stack([torch.zeros(14),torch.ones(14)]),
     'prompt':['Move the pill bottle to the pad.']*2,'actions':commands}
processed=model.input_transform(raw,transpose=False)
assert processed['tokenized_prompt'].shape==(2,200)
assert not torch.equal(processed['tokenized_prompt'][0],processed['tokenized_prompt'][1])
batch={k:v for k,v in raw.items() if k.startswith('observation/')}
batch.update(action=commands.flatten(1),tokenized_prompt=processed['tokenized_prompt'],tokenized_prompt_mask=processed['tokenized_prompt_mask'])
prepared=model.prepare_dagger_sft_batch(batch)
torch.testing.assert_close(prepared['actions'],processed['actions'].to(torch.float32),rtol=0,atol=0)
assert torch.equal(prepared['observation'].tokenized_prompt,processed['tokenized_prompt'])
back=model.output_transform({'actions':prepared['actions'],'state':processed['state']})
torch.testing.assert_close(back['actions'],commands.to(back['actions'].dtype),rtol=1e-5,atol=1e-6)
assert prepared['actions'].shape==(2,50,32)
assert all(x.shape==(2,3,224,224) for x in prepared['observation'].images.values())
result={'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),
    'fixed_seeds':fixed,'sidney_fixed_seeds':sidney_fixed,'identical_fixed32':True,
    'seed_input':seed_input,'seed_file_sha256':hashlib.sha256(seed_text.encode()).hexdigest(),
    'prepared_action_shape':list(prepared['actions'].shape),'tokens_shape':list(processed['tokenized_prompt'].shape),
    'state_changes_tokens':True,'stored_tokens_preserved':True,'action_roundtrip_max_error':(back['actions']-commands).abs().max().item(),
    'model_forward_or_update':False,'gpu_allocated':False,
    'scope':'Real local norm/tokenizer/input-output transforms and CPU preparation; native pi05 SFT/FSDP/update/save tested by subsequent smoke.'}
(packet/'smoke-resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
(packet/'delta.json').write_text(json.dumps(delta,indent=2))
(packet/'validation.json').write_text(json.dumps(result,indent=2))
print('PI05_BC_VALIDATION',json.dumps(result),flush=True)
