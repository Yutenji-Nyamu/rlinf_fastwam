"""Read-only, CPU-only small DSRL-network and replay audit; no VLA forward."""
import os,json,importlib.util,hashlib,time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import torch
torch.set_num_threads(2)
torch.manual_seed(1105)
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/parallel-dsrl-u-20261104')
def module(rel):
 spec=importlib.util.spec_from_file_location(Path(rel).stem,R/rel);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
c=module('rlinf/models/embodiment/modules/compact_encoders.py')
g=module('rlinf/models/embodiment/modules/gaussian_policy.py')
roots={'rt_clean':'/data/chenyiteng/results/robotwin-dsrl-full-20261104/clean','dojo_clean':'/data/chenyiteng/results/dojo-offpolicy/dojo-dsrl-full-g5-20261104','dojo_u':'/data/chenyiteng/results/dojo-offpolicy/dojo-dsrl-u-g6-20261104','rt_u':'/data/chenyiteng/results/robotwin-dsrl-full-20261104/u-g7'}
def stat(x):
 x=x.detach().float();return {'mean':x.mean().item(),'std':x.std().item(),'min':x.min().item(),'max':x.max().item()}
out={'time':time.time(),'cpu_only':True,'runs':{}}
for name,root in roots.items():
 cs=sorted(Path(root).glob('formal/**/checkpoints/global_step_*'),key=lambda x:int(x.name.split('_')[-1]),reverse=True)
 cp=next(x for x in cs if (x/'actor/local_shard_checkpoint/checkpoint_rank_0.pt').exists() and (x/'actor/sac_components/dsrl_trainer_state_rank_0.pt').exists())
 ck=torch.load(cp/'actor/local_shard_checkpoint/checkpoint_rank_0.pt',map_location='cpu',mmap=True,weights_only=False)
 replay=torch.load(cp/'actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt',map_location='cpu',mmap=True,weights_only=False)
 trainer=torch.load(cp/'actor/sac_components/dsrl_trainer_state_rank_0.pt',map_location='cpu',weights_only=False)
 models={'actor_image_encoder':c.LightweightImageEncoder64(),'actor_state_encoder':c.CompactStateEncoder(14,64),'dsrl_action_noise_net':g.GaussianPolicy(128,1600,action_horizon=1),'critic_image_encoder':c.LightweightImageEncoder64(),'critic_state_encoder':c.CompactStateEncoder(14,64),'q_head':c.CompactMultiQHead(64,64,1600)}
 for key,model in models.items():
  model.load_state_dict({k[len(key)+1:]:v for k,v in ck['model'].items() if k.startswith(key+'.')});model.eval()
 n=replay['resident_size'];s=replay['storage'];b=256
 idx=torch.linspace(max(0,n-2048),n-1,b).long()
 images=s['curr_obs']['main_images'][idx].float().unsqueeze(1)
 states=s['curr_obs']['states'][idx].bfloat16().float()
 actions=s['actions'][idx].float()
 with torch.no_grad():
  aimg=models['actor_image_encoder'](images);ast=models['actor_state_encoder'](states)
  qimg=models['critic_image_encoder'](images);qst=models['critic_state_encoder'](states)
  policy=models['dsrl_action_noise_net'];feat=torch.cat([ast,aimg],-1)
  h=policy.shared_net(feat);mu=policy.mean_layer(h);ls=policy.log_std_layer(h).clamp(-20,2)
  pi,lp=policy.sample(feat);pi=pi.flatten(1)
  q=models['q_head'];qdata=q(qst,qimg,actions);qpi=q(qst,qimg,pi)
  shuffled=torch.randperm(b)
  q_shufobs=q(qst[shuffled],qimg[shuffled],actions)
  q_shufaction=q(qst,qimg,actions[shuffled])
  q_zero=q(qst,qimg,torch.zeros_like(actions))
  q_gauss=q(qst,qimg,torch.randn_like(actions))
  q_tanh=q(qst,qimg,torch.randn_like(actions).tanh())
  phase0=min(2000 if name.startswith('rt') else 500,n)
  phase0a=s['actions'][:phase0].float();recenta=actions
  row={'checkpoint':str(cp),'update_step':trainer['update_step'],'resident':n,'reward_counts':{str(x):int((s['rewards'][:n]==x).sum()) for x in [-1,0,1]},'terminations':int(s['terminations'][:n].sum()),'truncations':int(s['truncations'][:n].sum()),'action_early':stat(phase0a),'action_recent':stat(recenta),'current_action':stat(pi),'pre_tanh_mean':stat(mu),'pre_tanh_std':stat(ls.exp()),'policy_entropy':float(-lp.mean()),'policy_mean_variation_across_obs':float(mu.std(0).mean()),'pi_saturated_fraction':float((pi.abs()>.95).float().mean()),'q_data':stat(qdata.mean(-1)),'q_policy':stat(qpi.mean(-1)),'q_zero':stat(q_zero.mean(-1)),'q_random_gaussian':stat(q_gauss.mean(-1)),'q_random_tanh':stat(q_tanh.mean(-1)),'q_obs_shuffle_mae':float((q_shufobs-qdata).abs().mean()),'q_action_shuffle_mae':float((q_shufaction-qdata).abs().mean()),'q_heads_disagreement':float(qdata.std(-1).mean()),'obs_image_across_batch_std':float(images.std(0).mean()),'image_feature_across_batch_std':float(qimg.std(0).mean()),'policy_image_feature_across_batch_std':float(aimg.std(0).mean()),'q_state_feature_across_batch_std':float(qst.std(0).mean()),'optimizer':[]}
  for opt in ck['optimizers']:
   row['optimizer'].append({'groups':[{k:v for k,v in pg.items() if k!='params'} for pg in opt['param_groups']],'first_state':{str(k):stat(v) if isinstance(v,torch.Tensor) and v.numel()>1 else str(v) for k,v in next(iter(opt['state'].values())).items()},'state_dtypes':sorted(set(str(v.dtype) for st in opt['state'].values() for k,v in st.items() if isinstance(v,torch.Tensor) and k!='step'))})
  row['trainable_parameters']={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
 out['runs'][name]=row
 del ck,replay,trainer,models
print(json.dumps(out))
