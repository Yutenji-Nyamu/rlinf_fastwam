import os,json,time,sys
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261015/expo-clean-entropy-b64-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'));from common import read,same,gpus,proc
q=read(T/'request.json');R=Path(q['run']);rt=Path(q['runtime']);launch=read(T/'launched.json');p=read(S/'plan.json');st=read(S/'status.json');before=read(T/'cutover-started.json')['plan_before']
o=dict(time=time.time(),driver_alive=same(launch['identity']),identity=launch['identity'],owner_alive=same(read(S/'owner-identity.json')),pair_owner_alive=same(launch['pair_owner']),old_exit=read(T/'old-released.json')['exit_code'],other_slots_preserved=all(p['slots'][g]==before['slots'][g] for g in ['6','7']),other_drivers_alive=all(same(p['slots'][g]['external_identity'])for g in ['6','7']),queue={})
for g in ['4','5']:
 f=read(p['slots'][g]['fallback']);cp=Path(f['checkpoint']);fr=Path(f['runtime']);o['queue'][g]=dict(phase=st['slots'][g]['phase'],request=p['slots'][g]['fallback'],unchanged=p['slots'][g]['fallback']==before['slots'][g]['fallback'],checkpoint=f['checkpoint'],complete=read(cp/'actor/sac_components/rlt_trainer_state/complete.json')['complete'],config=(fr/'resolved.yaml').is_file(),environment=(fr/'environment.json').is_file(),not_dispatched=not(fr/'started.json').exists())
for name in ['status.json','failure.json','complete.json']:
 if(R/'run'/name).exists():o[name]=read(R/'run'/name)
for name in ['status.json','released.json']:
 if(rt/name).exists():o['runtime_'+name]=read(rt/name)
o['gpu']={g:dict(v,processes=[dict(x,identity=proc(x['pid']))for x in v['processes']])for g,v in gpus().items()}
path=rt/'driver.log'
with path.open('rb')as f:f.seek(max(0,path.stat().st_size-2500));o['log_tail']=f.read().decode(errors='replace')
o['events_tail']=[]
if(R/'run/events.jsonl').exists():
 with(R/'run/events.jsonl').open('rb')as f:f.seek(max(0,(R/'run/events.jsonl').stat().st_size-30000));lines=f.read().decode(errors='replace').splitlines()
 for line in lines:
  try:v=json.loads(line)
  except ValueError:continue
  o['events_tail'].append({k:v[k]for k in ['time','event','episode','call','physical_steps']if k in v})
o['events_tail']=o['events_tail'][-12:]
(T/'latest-check.json').write_text(json.dumps(o,indent=2));print(json.dumps(o))
