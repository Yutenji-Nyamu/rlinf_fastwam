"""CPU-only audit of saved optimizer arithmetic; never writes checkpoints."""
import os,json,torch
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
torch.set_num_threads(2)
roots={'rt_clean':'/data/chenyiteng/results/robotwin-dsrl-full-20261104/clean','dojo_clean':'/data/chenyiteng/results/dojo-offpolicy/dojo-dsrl-full-g5-20261104'}
out={}
for name,root in roots.items():
 cs=sorted(Path(root).glob('formal/**/checkpoints/global_step_*'),key=lambda x:int(x.name.split('_')[-1]),reverse=True)
 cp=next(x for x in cs if (x/'actor/local_shard_checkpoint/checkpoint_rank_0.pt').exists())
 ck=torch.load(cp/'actor/local_shard_checkpoint/checkpoint_rank_0.pt',map_location='cpu',mmap=True,weights_only=False)
 groups=[['dsrl_action_noise_net','actor_image_encoder','actor_state_encoder'],['critic_image_encoder','critic_state_encoder','q_head']]
 rows=[]
 for opt,prefixes in zip(ck['optimizers'],groups):
  ordered=[(k,v) for k,v in ck['model'].items() if k.split('.')[0] in prefixes]
  pg=opt['param_groups'][0];ids=pg['params'];assert len(ids)==len(ordered),(len(ids),len(ordered))
  blocks={}
  for pid,(key,p) in zip(ids,ordered):
   st=opt['state'][pid];m,v=st['exp_avg'],st['exp_avg_sq'];assert m.shape==p.shape,(key,m.shape,p.shape)
   t=float(st['step']);b1,b2=pg['betas'];mf=m.float()/(1-b1**t);vf=v.float()/(1-b2**t)
   delta=pg['lr']*mf/(vf.sqrt()+pg['eps'])
   rounded=(p.float()-delta).bfloat16()
   block=key.split('.')[0];z=blocks.setdefault(block,{'n':0,'zero_quantized_updates':0,'nonzero_moment':0,'variance_no_decay':0,'delta_sum':0.0})
   nonzero=m!=0;z['n']+=p.numel();z['zero_quantized_updates']+=int(((rounded==p)&nonzero).sum());z['nonzero_moment']+=int(nonzero.sum());z['variance_no_decay']+=int(((v.float()*b2).bfloat16()==v).sum());z['delta_sum']+=float(delta.abs().sum())
  for key,z in blocks.items():
   z['rounds_to_no_weight_change_fraction']=z['zero_quantized_updates']/max(z['nonzero_moment'],1);z['variance_decay_rounds_away_fraction']=z['variance_no_decay']/z['n'];z['estimated_abs_update_mean']=z['delta_sum']/z['n']
  rows.append({'betas':pg['betas'],'lr':pg['lr'],'blocks':blocks})
 out[name]={'cp':str(cp),'assumption':'last stored Adam moments, bias correction, before next gradient; rounding diagnostic not measured full-run lost updates','optimizers':rows}
print(json.dumps(out))
