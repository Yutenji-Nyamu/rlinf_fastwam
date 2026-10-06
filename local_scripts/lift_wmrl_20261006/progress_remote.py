"""Read completed stages, scalar series and recent service progress only."""
import datetime,json,os,socket
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
O=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/runs/lift-pot-v1')
out={'time':datetime.datetime.now().astimezone().isoformat(),'receipts':{},'logs':{},'tensorboard':{},'services':{}}
for rel in ['error.json','recovery-error.json','prechecks.json','native_lift32/summary.json','startup_smoke/result.json','startup_smoke/verified-placement.json','formal/driver-identity.json','formal/driver-finished.json','formal/result.json','final.json']:
 p=O/rel
 if p.exists():out['receipts'][rel]=json.loads(p.read_text())
for rel in ['startup_smoke/driver.log','formal/driver.log']:
 p=O/rel
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-120000));lines=f.read().decode(errors='replace').splitlines()
  out['logs'][rel]=[x for x in lines if any(k in x.lower() for k in ['epoch','grad','success','reward','loss','traceback','error','checkpoint'])][-20:]
for stage in ['startup_smoke','formal']:
 paths=list((O/stage).glob('**/events.out.tfevents.*'))
 for p in paths[:8]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  e=EventAccumulator(str(p),size_guidance={'scalars':0});e.Reload()
  out['tensorboard'][str(p.relative_to(O))]={}
  for tag in e.Tags()['scalars']:
   if any(x in tag.lower() for x in ['success','grad','reward','loss','valid','advantage','time','step']):
    vals=e.Scalars(tag);out['tensorboard'][str(p.relative_to(O))][tag]={'count':len(vals),'last':[{'step':v.step,'value':v.value,'wall_time':v.wall_time} for v in vals[-5:]]}
for key in ['wm6','wm7']:
 p=O/'services'/key/'records/service-events.jsonl'
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-160000));lines=f.read().decode(errors='replace').splitlines()
  rows=[]
  for line in lines:
   try:rows.append(json.loads(line))
   except ValueError:pass
  out['services'][key]=[r for r in rows if r.get('event')!='heartbeat'][-5:]
print(json.dumps(out))
