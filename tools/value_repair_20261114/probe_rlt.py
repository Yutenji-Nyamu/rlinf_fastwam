"""CPU-only fixed-replay critic repair; never modifies the source checkpoint."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TF='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
import argparse,copy,json,sys,time,tempfile
from pathlib import Path
import torch
torch.set_num_threads(4);torch.manual_seed(1114)
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=2000);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--samples-per-buffer',type=int,default=4096);p.add_argument('--next-action',choices=['reference','actor'],default='actor');p.add_argument('--reset-critic',action='store_true');a=p.parse_args()
start=time.time();D=Path(a.output);D.mkdir(parents=True,exist_ok=True);cp=Path(a.checkpoint)
sys.path.insert(0,a.source)
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
cfg=json.loads(Path(a.config).read_text())
kw={k:v for k,v in cfg['actor']['model'].items() if k in ['z_dim','proprio_dim','action_dim','num_action_chunks','ref_num_action_chunks','add_q_head','q_head_type','fixed_std','output_activation']}
m=RLTMLPPolicy(**kw).eval();fresh=copy.deepcopy(m.q_head.state_dict())
state=torch.load(cp/'actor/model_state_dict/full_weights.pt',map_location='cpu',weights_only=False);m.load_state_dict(state,strict=True)
t=RLTMLPPolicy(**kw).eval();t.load_state_dict(torch.load(cp/'actor/sac_components/target_model/checkpoint_rank_0.pt',map_location='cpu',weights_only=False),strict=True)
for name,param in m.named_parameters():param.requires_grad_(name.startswith('q_head.'))
for param in t.parameters():param.requires_grad_(False)
opt=torch.optim.Adam(m.q_head.parameters(),lr=cfg['actor']['critic_optim']['lr'])
out={'checkpoint':str(cp),'config':a.config,'next_action':a.next_action,'updates':a.updates,'optimizer':'fresh','reset_critic':a.reset_critic,'scope':'fixed replay; CPU; only Q updated; no simulator/VLA inference'}

# RLinf DCP contains a merged optimizer state keyed by unflattened parameter FQN.
try:
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    with tempfile.TemporaryDirectory(dir=D) as temp:
        dest=Path(temp)/'dcp.pt';dcp_to_torch_save(str(cp/'actor/dcp_checkpoint'),str(dest))
        saved=torch.load(dest,map_location='cpu',weights_only=False)
    def locate(v):
        if isinstance(v,dict):
            if 'optimizers' in v:return v['optimizers']
            for x in v.values():
                found=locate(x)
                if found is not None:return found
        return None
    optim=locate(saved);assert optim is not None
    names=dict(m.named_parameters());groups=[]
    for group in optim['param_groups']:
        selected=[name for name in group['params'] if name.startswith('q_head.')]
        if selected:
            new={k:v for k,v in group.items() if k!='params'};new.update(params=[names[n] for n in selected],fused=False,capturable=False);groups.append(new)
    assert sum(len(g['params']) for g in groups)==len(list(m.q_head.parameters()))
    opt=torch.optim.Adam(groups)
    for name,v in optim['state'].items():
        if name.startswith('q_head.'):
            opt.state[names[name]]={k:x.detach().cpu().clone() if torch.is_tensor(x) else copy.deepcopy(x) for k,x in v.items()}
    out['optimizer']='restored named Q Adam state from DCP';out['optimizer_parameter_states']=len(opt.state)
except Exception as exc:
    out['optimizer_restore_error']=type(exc).__name__+': '+str(exc)
    raise RuntimeError('Exact Q optimizer restore failed; investigate before comparing') from exc

buffers=[]
for folder in ['reference_replay','replay_buffer']:
    files=sorted((cp/f'actor/sac_components/{folder}/rank_0').glob('trajectory_*.pt'),key=lambda f:int(f.name.split('_')[1]))
    assert files,folder
    ids=sorted(set(int(i) for i in torch.linspace(0,len(files)-1,min(a.samples_per_buffer,len(files))).tolist()))
    samples=[torch.load(files[i],map_location='cpu',weights_only=False) for i in ids]
    def obs(key):return {k:torch.cat([x[key][k].reshape(1,-1) for x in samples]) for k in ['z_rl','proprio','ref_chunk']}
    b={'obs':obs('curr_obs'),'next':obs('next_obs')}
    for k in ['actions','rewards','terminations','truncations']:b[k]=torch.cat([x[k].reshape(1,-1) for x in samples])
    b['heldout']=torch.arange(len(samples))%5==0
    buffers.append(b)
def batch(ids_by_buffer):
    d={}
    for k in buffers[0]:
        if k=='heldout':continue
        if isinstance(buffers[0][k],dict):d[k]={name:torch.cat([b[k][name][ids] for b,ids in zip(buffers,ids_by_buffer)]) for name in buffers[0][k]}
        else:d[k]=torch.cat([b[k][ids] for b,ids in zip(buffers,ids_by_buffer)])
    return d
held=batch([b['heldout'].nonzero().flatten() for b in buffers]);train_ids=[(~b['heldout']).nonzero().flatten() for b in buffers]
gamma=cfg['algorithm']['gamma'];tau=cfg['algorithm']['tau'];C=held['rewards'].shape[1]
discounts=torch.tensor([gamma**i for i in range(C)])
def target(b):
    with torch.no_grad():
        na=b['next']['ref_chunk'] if a.next_action=='reference' else m.sac_forward(b['next'])[0]
        raw=(b['rewards']*discounts).sum(-1)+(~b['terminations'].any(-1))*gamma**C*t.sac_q_forward(b['next'],na).min(-1).values
        return raw.clamp(0,1),raw
def evaluate():
    with torch.no_grad():
        qs=m.sac_q_forward(held['obs'],held['actions']);q=qs.mean(-1);y,raw=target(held)
        pos=held['rewards'].gt(0).any(-1);neg=~pos
        qp=q[pos];qn=q[neg]
        return {'q_mean':float(q.mean()),'q_min':float(q.min()),'q_max':float(q.max()),'q_outside_fraction':float(((q<0)|(q>1)).float().mean()),'target_raw_mean':float(raw.mean()),'target_clipped_fraction':float((raw!=y).float().mean()),'bounded_td_rmse':float((qs-y[:,None]).square().mean().sqrt()),'positive_terminal_count':int(pos.sum()),'other_count':int(neg.sum()),'q_positive_terminal':float(qp.mean()),'q_other':float(qn.mean()),'immediate_success_auc':float(((qp[:,None]>qn[None,:]).float()+.5*(qp[:,None]==qn[None,:]).float()).mean()),'ranking_scope':'immediate terminal reward vs other transitions; not full future-return calibration'}
out['before_original']=evaluate()
if a.reset_critic:
    m.q_head.load_state_dict(fresh);t.q_head.load_state_dict(fresh);opt.state.clear()
for model in (m,t):
    for head in model.q_head.qs:head.output_feature_normalization=True
out['before_repair']=evaluate();out['trace']=[]
for step in range(a.updates):
    ids=[train[torch.randint(len(train),(a.batch_size//2,))] for train in train_ids];b=batch(ids)
    y,_=target(b);opt.zero_grad(set_to_none=True)
    loss=(m.sac_q_forward(b['obs'],b['actions'])-y[:,None]).square().mean();loss.backward()
    torch.nn.utils.clip_grad_norm_(m.q_head.parameters(),cfg['actor']['critic_optim']['clip_grad']);opt.step()
    with torch.no_grad():
        for pt,pm in zip(t.q_head.parameters(),m.q_head.parameters()):pt.lerp_(pm,tau)
    if (step+1)%250==0 or step+1==a.updates:
        out['trace'].append({'update':step+1,'elapsed':time.time()-start,**evaluate()});(D/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['trace'][-1]),flush=True)
out['after']=evaluate();out['actor_unchanged']=all(torch.equal(v,m.state_dict()[k]) for k,v in state.items() if not k.startswith('q_head.'));assert out['actor_unchanged']
out['elapsed']=time.time()-start
torch.save({'q_head':m.q_head.state_dict(),'target_q_head':t.q_head.state_dict(),'critic_optimizer':opt.state_dict(),'note':'diagnostic result only; not a deployable RLinf checkpoint'},D/'critic-probe.pt')
(D/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
