"""Fixed Control seeds, four environments, success-once over 200 real actions.

Evaluation never inserts replay or changes training action/cadence counters.
The trainer owns RNG checkpoint/restore around this isolated evaluation hook.
"""
from __future__ import annotations
import copy,hashlib,json,os,time,uuid
from pathlib import Path
import torch
from omegaconf import OmegaConf,open_dict
from .backend import create_robotwin_env

def _atomic(path,data):
 temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with temporary.open('x') as stream:
  json.dump(data,stream,indent=2);stream.flush();os.fsync(stream.fileno())
 os.replace(temporary,path)

def evaluate(backend,learner,eval_config,run,seed_path,base_only=False,heartbeat=None):
 cfg=copy.deepcopy(OmegaConf.to_container(eval_config,resolve=True) if OmegaConf.is_config(eval_config) else eval_config)
 expected_seed_sha=cfg.pop('_expo_seed_sha256')
 seed_bytes=Path(seed_path).read_bytes();seed_sha=hashlib.sha256(seed_bytes).hexdigest()
 if seed_sha!=expected_seed_sha:raise ValueError('Fixed Control evaluation seed bytes drifted')
 task=cfg['task_config']['task_name'];seeds=json.loads(seed_bytes)[task]['success_seeds']
 if len(seeds)!=20 or len(set(seeds))!=20 or any(type(seed) is not int for seed in seeds):
  raise ValueError('Evaluation requires exactly twenty established unique Control seeds')
 if (cfg['total_num_envs']!=4 or cfg['rollout_epoch']!=5 or cfg['max_episode_steps']!=200 or
     cfg['max_steps_per_rollout_epoch']!=200 or cfg['task_config']['step_lim']!=200 or
     not cfg['ignore_terminations'] or not cfg['is_eval']):
  raise ValueError('Fixed Control 4x5, 200-action evaluation contract changed')
 contract={'task':task,'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
  'seed_sha256':seed_sha,'seeds':seeds,'policy':'raw-base' if base_only else 'EXPO',
  'base_updates':backend.base_updates,'learner_calls':learner.update_calls,'steps_per_episode':200,
  'parallel_envs':4,'n_base':1 if base_only else 8,'eval_noise_seed':90042}
 run=Path(run);run.mkdir(parents=True,exist_ok=True)
 complete=run/'complete.json'
 if complete.exists():
  result=json.loads(complete.read_text())
  if result.get('contract')!=contract or result.get('ok') is not True:raise ValueError('Prior eval receipt differs')
  return result
 attempt=run/('attempt-'+str(uuid.uuid4()));attempt.mkdir()
 cfg['task_config']['save_path']=str(attempt/'env-data')
 cfg['video_cfg']['video_base_dir']=str(attempt/'video')
 env=None;started=time.time();rows=[]
 generator=torch.Generator(device=backend.device).manual_seed(90042)
 learner.generator.manual_seed(90043)
 def beat():
  if heartbeat:heartbeat()
 def event(name,**fields):
  with (attempt/'events.jsonl').open('a') as stream:stream.write(json.dumps({'time':time.time(),'event':name,**fields})+'\n')
 try:
  beat();env=create_robotwin_env(OmegaConf.create(cfg),num_envs=4,seed_offset=0)
  # The validated training factory explicitly clears ignore_terminations.
  # Restore the exact Control evaluation wrapper flag after construction.
  env.ignore_terminations=True
  with open_dict(env.cfg):env.cfg.ignore_terminations=True
  event('initialized',contract=contract,renderer=env.expo_renderer_binding)
  for group in range(5):
   selected_seeds=seeds[group*4:(group+1)*4]
   obs,_=env.reset(env_seeds=selected_seeds);success=torch.zeros(4,dtype=torch.bool)
   event('group_started',group=group,seeds=selected_seeds)
   for chunk in range(20):
    beat()
    base=backend.sample_normalized(obs,num_candidates=1 if base_only else 8,generator=generator)[:,:,:10,:14]
    actions=base[:,0] if base_only else learner.select_actions(backend.critic_observation(obs),base)['actions']
    canonical=backend.decode(obs,actions)
    for step in range(10):
     obs,reward,term,trunc,info=env.step(canonical[:,step:step+1,:],auto_reset=False)
     reported=info.get('episode',{}).get('success_once',info.get('success'))
     if reported is None:raise ValueError('Native Control success-once metric missing')
     if isinstance(reported,list):reported=torch.as_tensor(reported)
     reported=torch.as_tensor(reported).detach().cpu().bool().reshape(-1)
     if reported.shape!=(4,):raise ValueError('Evaluation success metric batch changed')
     success|=reported
     elapsed=chunk*10+step+1
     if elapsed<200 and torch.as_tensor(trunc).any():raise RuntimeError('Evaluation ended before the inherited 200-action horizon')
    event('chunk',group=group,actions_per_env=(chunk+1)*10,success_once=success.tolist())
    beat()
   rows.extend({'seed':seed,'success_once':bool(ok),'physical_actions':200} for seed,ok in zip(selected_seeds,success.tolist()))
   event('group_finished',group=group,rows=rows[-4:]);beat()
 finally:
  if env is not None:env.offload()
  beat()
 result={'ok':True,'contract':contract,'rows':rows,'successes':sum(row['success_once'] for row in rows),
  'episodes':20,'success_rate':sum(row['success_once'] for row in rows)/20,
  'evaluation_physical_actions':4000,'training_budgeted_actions':0,'elapsed_seconds':time.time()-started,
  'attempt':str(attempt),'env_closed':True,'metric':'Control eval/success_once; term intentionally ignored'}
 _atomic(complete,result);return result
