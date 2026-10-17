import os,sys,time,json,tarfile,statistics
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
T=Path('/data/chenyiteng/deployment-20261017/pi05-policy-budget-eval-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,same
p=read(T/'prepared.json');R=Path(p['old_run']);rt=R/'runtime'
assert (T/'cutover-complete.json').exists() and not same(p['old_identity'])
assert not (T/'grpo-u-tau1-closeout.json').exists()
curves={};eventfiles=sorted(R.rglob('events.out.tfevents.*'))
for parent in sorted(set(f.parent for f in eventfiles)):
 a=EventAccumulator(str(parent),size_guidance={'scalars':0});a.Reload()
 for tag in a.Tags().get('scalars',[]):curves.setdefault(tag,[]).extend([[v.step,v.value,v.wall_time] for v in a.Scalars(tag)])
curves={k:sorted({v[0]:v for v in vals}.values()) for k,vals in curves.items()}
save(T/'grpo-u-tau1-curves.json',curves)
checkpoints=[]
for cp in sorted((R/R.name/'checkpoints').glob('global_step_*'),key=lambda p:int(p.name.split('_')[-1])):
 weights=[f for f in cp.rglob('*.pt') if f.is_file() and f.stat().st_size>2**30]
 if weights:checkpoints.append(dict(path=str(cp),step=int(cp.name.split('_')[-1]),files=[dict(path=str(f),bytes=f.stat().st_size) for f in weights]))
rows=curves['eval/success_once'];completed=len(curves['env/success_once'])
data=dict(time=time.time(),run=str(R),method='grpo-u',tau=1,gpus=[6,7],completed_updates=completed,planned_updates=200,completed_budget=completed>=200,reason='User-authorized closeout; reallocate6/7 to paired policy and inference budget evaluation.',release=read(T/'grpo-released.json'),first_eval=rows[0],latest_eval=rows[-1],first5_mean=statistics.mean(v[1] for v in rows[:5]),last5_mean=statistics.mean(v[1] for v in rows[-5:]),peak_eval=max(v[1] for v in rows),checkpoints=checkpoints,latest_checkpoint=checkpoints[-1])
files=[(T/'grpo-u-tau1-curves.json','curves.json'),(T/'grpo-stop-requested.json','stop-requested.json'),(T/'grpo-released.json','released.json')]
for f in eventfiles:files.append((f,str(f.relative_to(R))))
for f in rt.iterdir():
 if f.is_file() and f.suffix in ('.json','.yaml','.txt'):files.append((f,'runtime/'+f.name))
log=rt/'driver.log';data['server_full_log']=dict(path=str(log),bytes=log.stat().st_size)
if log.stat().st_size<50*1024**2:files.append((log,'runtime/driver.log'))
else:
 with log.open('rb') as f:f.seek(-8*1024**2,2);(T/'grpo-driver-tail.log').write_bytes(f.read())
 files.append((T/'grpo-driver-tail.log','runtime/driver-tail.log'))
archive=T/'grpo-u-tau1-closeout-logs.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for path,arc in files:t.add(path,arcname=arc,recursive=False)
data['archive']=dict(path=str(archive),bytes=archive.stat().st_size,members=len(files))
save(T/'grpo-u-tau1-closeout.json',data)
print(json.dumps({k:data[k] for k in ['completed_updates','first5_mean','last5_mean','latest_eval','peak_eval','archive','latest_checkpoint']}))
