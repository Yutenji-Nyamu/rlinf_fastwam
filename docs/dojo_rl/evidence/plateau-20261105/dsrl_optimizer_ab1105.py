"""One disposable CPU critic update with identical gradients: BF16 vs FP32."""
import os,json,importlib.util,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import torch
from pathlib import Path
torch.set_num_threads(2);torch.manual_seed(1105)
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/parallel-dsrl-u-20261104')
def mod(rel):
 s=importlib.util.spec_from_file_location(Path(rel).stem,R/rel);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=mod('rlinf/models/embodiment/modules/compact_encoders.py');g=mod('rlinf/models/embodiment/modules/gaussian_policy.py')
roots={'rt_clean':'/data/chenyiteng/results/robotwin-dsrl-full-20261104/clean','dojo_clean':'/data/chenyiteng/results/dojo-offpolicy/dojo-dsrl-full-g5-20261104'}
out={}
for name,root in roots.items():
 cp=next(x for x in sorted(Path(root).glob('formal/**/checkpoints/global_step_*'),key=lambda x:int(x.name.split('_')[-1]),reverse=True) if (x/'actor/local_shard_checkpoint/checkpoint_rank_0.pt').exists())
 ck=torch.load(cp/'actor/local_shard_checkpoint/checkpoint_rank_0.pt',map_location='cpu',mmap=True,weights_only=False)
 rp=torch.load(cp/'actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt',map_location='cpu',mmap=True,weights_only=False)
 ts=torch.load(cp/'actor/sac_components/dsrl_trainer_state_rank_0.pt',map_location='cpu',weights_only=False)
 def critic():return torch.nn.ModuleDict({'critic_image_encoder':c.LightweightImageEncoder64(),'critic_state_encoder':c.CompactStateEncoder(14,64),'q_head':c.CompactMultiQHead(64,64,1600)})
 q=critic();q.load_state_dict({k:v for k,v in ck['model'].items() if k.split('.')[0] in q})
 target=critic();target.load_state_dict({k.removeprefix('_fsdp_wrapped_module.'):v for k,v in ts['target_shadow_f32'].items()})
 ai=c.LightweightImageEncoder64();ast=c.CompactStateEncoder(14,64);ap=g.GaussianPolicy(128,1600,action_horizon=1)
 for key,m in [('actor_image_encoder',ai),('actor_state_encoder',ast),('dsrl_action_noise_net',ap)]:m.load_state_dict({k[len(key)+1:]:v for k,v in ck['model'].items() if k.startswith(key+'.')})
 n=rp['resident_size'];s=rp['storage'];idx=torch.randperm(n)[:64]
 def features(m,prefix):
  im=s[prefix]['main_images'][idx].float().unsqueeze(1);st=s[prefix]['states'][idx].bfloat16().float()
  return m['critic_state_encoder'](st),m['critic_image_encoder'](im)
 with torch.no_grad():
  ni=s['next_obs']['main_images'][idx].float().unsqueeze(1);ns=s['next_obs']['states'][idx].bfloat16().float()
  na,_=ap.sample(torch.cat([ast(ns),ai(ni)],-1));nf=features(target,'next_obs')
  y=s['rewards'][idx]+s['discounts'][idx]*s['continuations'][idx]*target['q_head'](*nf,na.flatten(1)).mean(-1,keepdim=True)
 current=q['q_head'](*features(q,'curr_obs'),s['actions'][idx].float())
 loss=(current-y).square().mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(q.parameters(),10)
 initial=[p.detach().clone() for p in q.parameters()];grad=[p.grad.detach().bfloat16().float().clone() for p in q.parameters()]
 variants={};savedopt=ck['optimizers'][1]
 for dtype in [torch.bfloat16,torch.float32]:
  model=copy.deepcopy(q).to(dtype);opt=torch.optim.Adam(model.parameters(),lr=.0003)
  opt.load_state_dict(copy.deepcopy(savedopt))
  for p,gr in zip(model.parameters(),grad):p.grad=gr.to(dtype)
  opt.step();stats={}
  for (key,p),before,gr in zip(model.named_parameters(),initial,grad):
   d=p.detach().float()-before;block=key.split('.')[0];z=stats.setdefault(block,{'n':0,'nonzero_grad':0,'unchanged_nonzero_grad':0,'sum_update2':0.0})
   nz=gr!=0;z['n']+=p.numel();z['nonzero_grad']+=int(nz.sum());z['unchanged_nonzero_grad']+=int(((d==0)&nz).sum());z['sum_update2']+=float(d.square().sum())
  for z in stats.values():z['unchanged_fraction']=z['unchanged_nonzero_grad']/max(z['nonzero_grad'],1);z['update_rms']=(z['sum_update2']/z['n'])**.5
  variants[str(dtype)]=stats
 out[name]={'cp':str(cp),'batch':64,'loss':float(loss),'grad_norm':float(norm),'fixed_same_quantized_gradient':True,'variants':variants}
print(json.dumps(out))
