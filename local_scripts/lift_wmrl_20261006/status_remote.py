"""Bounded live task evidence, with no environment or credential output."""
import datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-pot-v1';O=S/'runs/lift-pot-v1'
read=lambda p:json.loads(p.read_text())
out={'time':datetime.datetime.now().astimezone().isoformat(),'files':{}}
for root,names in [(D,['launch.json','prepared/ready.json','prepared/config-contract.json']),(O,['owner-identity.json','state.json','heartbeat.json','final.json','rlt-return-dispatched.json','native_lift32/summary.json','native_lift32/phase.json','cleanup.json'])]:
 for name in names:
  p=root/name
  if p.is_file():
   row=read(p)
   if name=='heartbeat.json':row.pop('live_processes',None)
   out['files'][str(p.relative_to(S))]=row
if (D/'launch.json').exists():
 row=read(D/'launch.json');p=Path('/proc')/str(row['pid'])
 out['owner_live']=False
 if p.exists():
  st=(p/'stat').read_text().split(') ',1)[1].split()
  out['owner_live']=p.stat().st_uid==20001 and int(st[19])==row['starttime'] and st[0] not in ['Z','X'] and hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()==row['cmdline_sha256']
for p in [D/'owner-console.log',O/'native_lift32/command-0.log',O/'native_lift32/command-1.log',O/'startup_smoke/driver.log',O/'formal/driver.log',O/'services/wm6/service.log',O/'services/wm7/service.log']:
 if p.is_file():out['files'][str(p.relative_to(S))]='\n'.join(p.read_text(errors='replace').splitlines()[-12:])
rows=[read(p) for p in (O/'native_lift32/capture').glob('native-*.json')]
for key in ['wm6','wm7']:
 for name in ['service-cpu-ready.json','records/service-manifest.json']:
  p=O/'services'/key/name
  if p.exists():out['files'][str(p.relative_to(S))]=read(p)
out['native_capture']={'count':len(rows),'success':sum(r['reference_success'] for r in rows),'unique_requested_seeds':len({r['seed'] for r in rows})}
out['gpu_summary']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
out['gpu_contexts']=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','m'],text=True,timeout=20)
print(json.dumps(out))
