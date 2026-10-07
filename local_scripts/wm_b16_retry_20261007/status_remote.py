"""Compact read-only state of the unique B16 formal owner."""
import json,os,socket,subprocess,urllib.request
from pathlib import Path
from datetime import datetime
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
F=S/'lift-two-gpu-b16-20261007-v1';O=S/'runs/lift-two-gpu-b16-20261007-v1'
read=lambda p:json.loads(Path(p).read_text())
def tail(p,n=4000):
 if not p.is_file():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
def same(i):
 try:
  p=Path('/proc')/str(i['pid']);st=(p/'stat').read_text().rsplit(')',1)[1].split()
  return p.stat().st_uid==20001 and int(st[19])==i['start'] and st[0] not in ['Z','X']
 except FileNotFoundError:return False
out={'time':datetime.now().astimezone().isoformat(),'files':{},'logs':{},'wm':{}}
for p in [F/'ready.json',F/'launch.json',O/'owner-identity.json',O/'state.json',O/'error.json',O/'final.json',
          O/'scope-activated.json',O/'recovery-error.json',O/'formal/driver-identity.json',O/'formal/verified-placement.json',
          O/'formal/result.json',O/'services/wm5/service-cpu-ready.json']:
 if p.is_file():
  r=read(p)
  if 'pid' in r and 'start' in r:r['alive_now']=same(r)
  out['files'][str(p.relative_to(S))]=r
for p in [F/'preparation.log',F/'owner-console.log',O/'formal/driver.log',O/'services/wm5/service.log']:
 if p.exists():out['logs'][str(p.relative_to(S))]=tail(p)
out['worker_proofs']=[]
for p in (O/'formal').glob('**/worker*.out'):
 t=tail(p,5000)
 for line in (t or '').splitlines():
  if any(s in line for s in ['WMRL_POLICY_ACTUAL_BATCH','WMRL_ACTOR_MEMORY','grad_norm','Epoch','OutOfMemory','Traceback']):out['worker_proofs'].append(line[-1600:])
p=O/'services/wm5/records/service-events.jsonl'
if p.is_file():
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];bs=[x for x in rows if x.get('event')=='batch_completed']
 keys=['timestamp_utc','actual_wm_batch','configured_wm_batch','seconds','wm_peak','reward_peak','cuda_allocated_bytes','cuda_reserved_bytes','outputs_finite']
 out['wm']={'batches':len(bs),'first':[{k:x[k] for k in keys if k in x} for x in bs[:2]],'last':[{k:x[k] for k in keys if k in x} for x in bs[-2:]],
            'phase':rows[-1].get('phase') if rows else None,'errors':[x.get('error') for x in rows if 'error' in x]}
try:
 with urllib.request.urlopen('http://127.0.0.1:18985/health',timeout=3) as r:out['health']=json.load(r)
except OSError:out['health']=None
cp=F/'cycles/cycle/plan.json'
if cp.is_file():
 out['rlt']={k:{'original_alive':same(r['original_identity']),'new_run':r['new_run'],'checkpoint':r['recovery']['checkpoint']['step']} for k,r in read(cp)['runs'].items()}
 out['borrowed']=(cp.parent/'rlt-stopped.json').exists()
out['protected']=[{'pid':p,'start':s,'alive_now':same({'pid':p,'start':s})} for p,s in [(3182779,727781893),(3182804,727782101),(3651109,727565004)]]
out['gpu']=subprocess.check_output(['nvidia-smi'],text=True,timeout=20)
print(json.dumps(out))
