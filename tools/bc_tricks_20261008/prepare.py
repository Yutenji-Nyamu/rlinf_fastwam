import copy,hashlib,json,os,sys,time
from pathlib import Path
import yaml
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'trick-ablation-20261008';sys.path.insert(0,str(S/'ops'))
from common import read,save,same,gpus
p=read(T/'plan-before.json');pre=read(T/'prestop.json');assert (T/'concluded.json').exists();assert not same(pre['owner']);assert all(same(q) for q in pre['protected']);host=p['host'];snap=gpus()
sys.path.insert(0,p['repo']);import torch
from rlinf.algorithms.online_bc_signal_tau import signal_tau_contract,apply_signal_tau

def replace(v,old,new):
 if isinstance(v,dict):return {k:replace(x,old,new) for k,x in v.items()}
 if isinstance(v,list):return [replace(x,old,new) for x in v]
 if isinstance(v,str):return new.name if v==old.name else v.replace(str(old),str(new))
 return v
new=[]
for r in pre['rows']:
 g=r['gpu'];q=r['q'];assert not snap[g]['processes'];old=Path(q['run']);kind=q['signal'];mode='none' if host=='sz1' and g in [4,5] else 'both' if host=='sz1' else 'drop' if host=='sz2' else 'anneal';tau=2.5 if mode=='none' else 3. if kind=='u' else 2.
 name=f'bc-{kind}-tau{tau:g}-{mode}-{host}-g{g}-300-1008-v1';run=Path('/data/chenyiteng/results/bc-tricks-20261008')/name;rt=run/'runtime';assert not run.exists()
 cfg=replace(yaml.safe_load((old/'runtime/resolved.yaml').read_text()),old,run);e=replace(read(old/'runtime/environment.json'),old,run);dv=cfg['algorithm']['online_bc']['dvac']
 dv['temperature_local']=dv['temperature_chunk']=tau;dv['chunk_dropout']['enabled']=mode in ['both','drop'];dv['alpha_schedule']['enabled']=mode in ['both','anneal']
 cfg['runner']['resume_dir']=None;assert cfg['algorithm']['update_epoch']==5 and cfg['actor']['global_batch_size']==1024
 ct=signal_tau_contract(dv,dv['signal_spec']);raw=torch.linspace(.01,.99,128*50).reshape(128,50);batch={ct['raw_key']:raw,'action_valid_mask':torch.ones(128,50,14)};state=torch.random.get_rng_state().clone();w,m=apply_signal_tau(batch,ct,runner_step=0,update_step=5);z,end=apply_signal_tau(batch,ct,runner_step=199,update_step=6)
 assert torch.equal(state,torch.random.get_rng_state()) and torch.isfinite(w).all() and m['final_weight_std']>0
 assert (m['dropout_fraction']>0)==dv['chunk_dropout']['enabled'];assert (end['alpha_local']==0)==dv['alpha_schedule']['enabled'];assert torch.equal(z,torch.ones_like(z)) if dv['alpha_schedule']['enabled'] else end['final_weight_std']>0
 for group in cfg['cluster']['node_groups']:
  for ec in group['env_configs']:ec['env_vars']=[{k:v} for k,v in sorted(e.items())]
 assert cfg['cluster']['component_placement']['actor,env,rollout']['placement']==str(g)
 assert Path(e['RLINF_OPENDW_GPU_SCOPE_MANIFEST']).is_file() and Path(e['LD_PRELOAD']).is_file()
 rt.mkdir(parents=True);(rt/'resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));save(rt/'environment.json',e)
 req=copy.deepcopy(q);req.update(run=str(run),runtime=str(rt),namespace=name,tau=tau,tricks=mode);rq=T/'requests'/f'bc-g{g}.json';save(rq,req);p['slots'][str(g)]['priority']=str(rq)
 new.append(dict(gpu=g,signal=kind,tau=tau,tricks=mode,run=str(run),request=str(rq),test=dict(first=m,last=end),source_head=r['source_head'],inherited_from=str(old)))
save(T/'plan-next.json',p);save(T/'prepared.json',dict(time=time.time(),host=host,rows=new,unchanged='Model,task,N8,U5,300 rounds,200 actions,GB1024,MB32,eval5x32,per-card RLT fallback; source unchanged',formal_stop='300 rounds or driver error; then same-card RLT'))
print(json.dumps(dict(host=host,prepared=[{k:r[k] for k in ['gpu','signal','tau','tricks','run']} for r in new],cpu_contract_checks='PASS')))
