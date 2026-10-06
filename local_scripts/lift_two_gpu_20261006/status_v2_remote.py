"""Compact live evidence for attempt 2 and the untouched GPU6/7 RLT drivers."""
import datetime,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-two-gpu-b32-v1';O=S/'runs/lift-two-gpu-b32-v2'
read=lambda p:json.loads(Path(p).read_text())
def tail(p,n=7000):
 if not p.is_file():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
out={'time':datetime.datetime.now().astimezone().isoformat(),'files':{},'rlt':{},'proofs':[]}
for name in ['owner-identity.json','state.json','startup-smoke-accepted.json','error.json','final.json','recovery-error.json','startup_smoke/result.json','formal/result.json','startup_smoke/verified-placement.json','formal/verified-placement.json','services/wm5/service-cpu-ready.json']:
 p=O/name
 if p.is_file():out['files'][name]=read(p)
for p in [D/'owner-v2-console.log',O/'startup_smoke/driver.log',O/'formal/driver.log',O/'services/wm5/service.log',O/'services/wm5/records/service-events.jsonl']:
 text=tail(p)
 if text is not None:out['files'][str(p.relative_to(S))]=text
for root in [O/'startup_smoke',O/'formal']:
 for p in root.rglob('*'):
  if p.suffix in ['.log','.out'] and p.stat().st_size<8000000:
   for line in p.read_text(errors='replace').splitlines():
    if any(x in line for x in ['WMRL_POLICY_','WMRL_ACTOR_MEMORY','WMRL_TWO_TO_ONE_RESTORE','Traceback (most recent','OutOfMemoryError']):
     out['proofs'].append({'file':str(p.relative_to(O)),'line':line[:12000]})
oldcp=read(S/'lift-pot-v1/prepared-cycles/cycle/plan.json')
for key in ['gpu6','gpu7']:
 row=oldcp['runs'][key];run=Path(row['new_run']);identity=read(run/'runtime/driver-identity.json');p=Path('/proc')/str(identity['pid'])
 alive=False
 if p.exists():
  st=(p/'stat').read_text().split(') ',1)[1].split();alive=p.stat().st_uid==20001 and int(st[19])==identity['start'] and st[0] not in ['Z','X']
 out['rlt'][key]={'pid':identity['pid'],'start':identity['start'],'alive':alive}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
out['contexts']=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','m'],text=True,timeout=20)
events=O/'services/wm5/records/service-events.jsonl'
if events.is_file():
 rows=[json.loads(x) for x in events.read_text().splitlines()]
 completed=[r for r in rows if r.get('event')=='batch_completed']
 out['wm_batches']={'count':len(completed),'recent':[{k:v for k,v in r.items() if k in ['timestamp_utc','actual_wm_batch','seconds','world_model_seconds','reward_seconds','wm_peak','reward_peak','rows_per_second','outputs_finite']} for r in completed[-8:]]}
print(json.dumps(out))
