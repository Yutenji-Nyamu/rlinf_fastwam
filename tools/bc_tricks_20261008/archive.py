import json,os,shutil,tarfile,time
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'trick-ablation-20261008';pre=json.loads((T/'prestop.json').read_text());assert (T/'concluded.json').exists();dest=T/'archive';dest.mkdir(exist_ok=True);out=[]
for r in pre['rows']:
 q=r['q'];root=Path(q['run']);pts={}
 for d,dirs,fs in os.walk(root):
  dirs[:]=[n for n in dirs if n not in ['checkpoints','success_data','video','robotwin_data','replay_buffer']]
  for n in fs:
   if not n.startswith('events.out.tfevents'):continue
   a=EventAccumulator(str(Path(d)/n),size_guidance={'scalars':0});a.Reload()
   for tag in a.Tags()['scalars']:
    for x in a.Scalars(tag):pts.setdefault(tag,{})[x.step]=[x.value,x.wall_time]
 cps=[]
 for cp in (root/root.name/'checkpoints').glob('global_step_*'):
  if all((cp/f).is_file() and (cp/f).stat().st_size>0 for f in ['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/online_bc/rank_0/learner.pt','actor/online_bc/rank_0/success_replay.pt','actor/online_bc/rank_0/signal_tau.pt']):cps.append(cp)
 cp=max(cps,key=lambda x:int(x.name.split('_')[-1]));ev=pts['eval/success_once'];on=pts['env/success_once'];last=max(on);metrics={tag:dict(step=max(v),value=v[max(v)][0],count=len(v)) for tag,v in pts.items() if v}
 record=dict(host=pre['host'],gpu=r['gpu'],run=root.name,source_head=r['source_head'],signal=q['signal'],tau=q['tau'],tricks='both',reason='USER_CONCLUDED',completed_rounds=last+1,checkpoint=str(cp),eval_round=max(ev)+1,eval_success=ev[max(ev)][0],last5eval_mean=sum(v[0] for k,v in sorted(ev.items())[-5:])/len(sorted(ev)[-5:]),metrics=metrics,curves={k:[[step,v[0],v[1]] for step,v in sorted(z.items())] for k,z in pts.items()})
 name=f'g{r["gpu"]}-{q["signal"]}-tau{q["tau"]}';(dest/(name+'.json')).write_text(json.dumps(record,indent=2));shutil.copyfile(root/'runtime/resolved.yaml',dest/(name+'.yaml'));out.append({k:v for k,v in record.items() if k not in ['metrics','curves']})
(T/'archive-summary.json').write_text(json.dumps(out,indent=2))
with tarfile.open(T/'archive.tar.gz','w:gz') as t:
 for f in dest.iterdir():t.add(f,arcname=f.name)
 t.add(T/'archive-summary.json',arcname='archive-summary.json');t.add(T/'concluded.json',arcname='concluded.json')
print(json.dumps(out));print('BYTES',(T/'archive.tar.gz').stat().st_size)
