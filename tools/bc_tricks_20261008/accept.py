import json,math,os,sys,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'trick-ablation-20261008';sys.path.insert(0,str(S/'ops'))
from common import read,save,same,gpus
p=read(S/'plan.json');x=read(T/'latest-probe.json');assert time.time()-x['time']<180;assert x['owner_alive'] and x['protected_unchanged'];snap=gpus();rows=[]
for r in x['rows']:
 q=read(p['slots'][r['gpu']]['priority']);m=r['metrics'];assert r['driver_alive'] and same(r['identity']) and r['finished'] is None and r['phase']=='PRIORITY_RUNNING'
 assert m['train/bc_signal_tau/runner_round']['value']>=1 and m['train/bc_signal_tau/final_weight_std']['value']>0
 assert m['train/bc/actor_loss']['value']>0 and m['train/actor/grad_norm']['value']>0
 assert all(math.isfinite(v['value']) for v in m.values())
 assert m['train/bc_signal_tau/temperature_local']['value']==q['tau']
 assert (m['train/bc_signal_tau/dropout_fraction']['value']>0)==(q['tricks'] in ['drop','both'])
 assert r['bindings'] and all(b['gpu']==int(r['gpu']) for b in r['bindings']);assert any(b['type'] in ['G','C+G'] for b in r['bindings'])
 rows.append({k:v for k,v in r.items() if k!='tail'})
pre=read(T/'prestop.json');assert all(same(q) for q in pre['protected'])
# Other users may freely occupy 0-3; only this namespace binding is constrained.
if not (S/'fallback-enabled.json').exists():save(S/'fallback-enabled.json',dict(time=time.time(),reason='New BC trick ablations passed real update; same-card RLT fallback retained',requests={g:s['fallback'] for g,s in p['slots'].items()}))
receipt=dict(time=time.time(),host=p['host'],owner=read(S/'owner-identity.json'),rows=rows,fallback_enabled=True,source='105170b87167a9f86b2d3c98ff2b74c4ec7025c3',gpu_0_3=snap|{} if False else {g:snap[g] for g in range(4)},protected_unchanged=pre['protected'])
save(T/'formal-accepted.json',receipt);save(S/'formal-accepted.json',receipt)
print(json.dumps(dict(host=p['host'],accepted=len(rows),rounds={r['gpu']:r['metrics']['train/bc_signal_tau/runner_round']['value'] for r in rows},fallback=True)))
