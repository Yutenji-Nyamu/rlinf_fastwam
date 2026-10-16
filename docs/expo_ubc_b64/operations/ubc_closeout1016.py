"""Archive explicit lightweight evidence after verified old-driver release."""
import os,json,sys,time,tarfile,hashlib,statistics,zipfile
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261016/expo-ubc-b64-tau123-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,same
assert (T/'cutover-complete.json').exists()
summaries=[]
for p in read(T/'prepared.json')['groups']:
 old=read(p['old_request']);R=Path(old['run']);rt=Path(old['runtime']);method='grpo-u' if old.get('namespace') else 'expo-editor-u';key=method+'-tau'+str(old['tau'])
 release=read(T/('tau'+str(p['tau'])+'-old-released.json'));assert not same(p['old_identity']) and release['gpu_clear']
 files=[(Path(p['old_request']),'request.json'),(Path(p['old_release']),'released.json')]
 data=dict(time=time.time(),run=str(R),method=method,tau=old['tau'],gpus=old['gpus'],release=release,reason='User-authorized early closeout to reallocate this pair to fresh EXPO B64 U-BC; not full-budget completion.',completed_budget=False)
 if method=='grpo-u':
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  curves={};eventfiles=sorted(R.rglob('events.out.tfevents.*'))
  for parent in sorted(set(f.parent for f in eventfiles)):
   a=EventAccumulator(str(parent),size_guidance={'scalars':0});a.Reload()
   for tag in a.Tags().get('scalars',[]):curves.setdefault(tag,[]).extend([[v.step,v.value,v.wall_time] for v in a.Scalars(tag)])
  curves={k:sorted({v[0]:v for v in vals}.values()) for k,vals in curves.items()}
  save(T/(key+'-curves.json'),curves);files.append((T/(key+'-curves.json'),'curves.json'))
  for f in eventfiles:files.append((f,str(f.relative_to(R))))
  for name in ['resolved.yaml','cleanup-targets.json','driver-identity.json','started.json']:
   files.append((rt/name,'runtime/'+name))
  cps=sorted((R/R.name/'checkpoints').glob('global_step_*'),key=lambda f:int(f.name.split('_')[-1]));checkpoints=[]
  for cp in cps:
   fs=[f for f in cp.rglob('*') if f.is_file()]
   weights=[f for f in fs if f.name in ['checkpoint_rank_0.pt','full_weights.pt']]
   if weights:checkpoints.append(dict(path=str(cp),step=int(cp.name.split('_')[-1]),files=[dict(path=str(f),bytes=f.stat().st_size) for f in weights]))
  latest=checkpoints[-1];assert all(zipfile.is_zipfile(f['path']) for f in latest['files'])
  rows=curves['eval/success_once'];data.update(latest_checkpoint=latest,checkpoints=checkpoints,completed_updates=len(curves['env/success_once']),first_eval=rows[0],latest_eval=rows[-1],first5_mean=statistics.mean(v[1] for v in rows[:5]),last5_mean=statistics.mean(v[1] for v in rows[-5:]),peak_eval=max(v[1] for v in rows))
 else:
  events=[]
  for line in (R/'run/events.jsonl').read_text().splitlines():
   try:events.append(json.loads(line))
   except ValueError:pass
  ev=[dict(path=str(f),result=read(f)) for f in sorted((R/'run/evaluations').glob('*/complete.json'))]
  rows=[v for v in ev if '/episode-' in v['path']];initial=[v for v in ev if '/initial-base/' in v['path']]
  weights=R/'run/checkpoint-latest.pt';assert zipfile.is_zipfile(weights)
  data.update(status=read(R/'run/status.json'),stopped=read(R/'run/stopped.json'),evaluations=ev,checkpoint=dict(path=str(weights),bytes=weights.stat().st_size,mtime=weights.stat().st_mtime,receipt=read(R/'run/checkpoint.json')),last_learning=[x for x in events if x.get('event')=='learner_finished'][-5:],events_tail=events[-6:],initial_raw_base=initial[0]['result']['success_rate'] if initial else None,first5_mean=statistics.mean(v['result']['success_rate'] for v in rows[:5]),last5_mean=statistics.mean(v['result']['success_rate'] for v in rows[-5:]),latest_eval=rows[-1]['result']['success_rate'],peak_eval=max(v['result']['success_rate'] for v in rows))
  for name in ['inputs.json','run/events.jsonl','run/status.json','run/stopped.json','run/resolved.json','run/checkpoint.json']:files.append((R/name,name))
  for f in (R/'run/evaluations').glob('*/complete.json'):files.append((f,str(f.relative_to(R))))
 log=rt/'driver.log';data['server_full_log']=dict(path=str(log),bytes=log.stat().st_size)
 if log.stat().st_size<=50*1024**2:files.append((log,'runtime/driver.log'))
 else:
  with log.open('rb') as stream:stream.seek(-8*1024**2,2);tail=stream.read()
  dest=T/(key+'-driver-tail.log');dest.write_bytes(tail);files.append((dest,'runtime/driver-tail.log'));data['archive_log_tail_bytes']=len(tail)
 archive=T/(key+'-closeout-logs.tar.gz');assert not archive.exists()
 with tarfile.open(archive,'w:gz') as tar:
  for path,arc in files:
   if path.is_file():tar.add(path,arcname=arc,recursive=False)
 data['archive']=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 save(T/(key+'-closeout.json'),data);summaries.append({k:v for k,v in data.items() if k in ['method','tau','run','completed_updates','latest_eval','first5_mean','last5_mean','peak_eval','archive','latest_checkpoint','checkpoint']})
print(json.dumps(summaries))
