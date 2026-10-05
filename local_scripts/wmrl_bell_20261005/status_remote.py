"""Read only the current owner and our GPU contexts; never clean any process."""
import datetime,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'click-bell-v2';O=S/'runs/click-bell-v2'
out=dict(time=datetime.datetime.now().astimezone().isoformat())
for key,path in [('launch',D/'launch.json'),('state',O/'state.json'),('identity',O/'owner-identity.json'),('error',O/'error.json'),('final',O/'final.json'),('recovery_error',O/'recovery-error.json'),('prechecks',O/'prechecks.json')]:
 if path.is_file():out[key]=json.loads(path.read_text())
out['logs']={}
paths=[D/'owner-console.log']+list(O.glob('services/*/service.log'))+list(O.glob('*/command-*.log'))+list(O.glob('*/driver.log'))
for p in paths:
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2200));raw=f.read()
  out['logs'][str(p.relative_to(S))]=raw.decode(errors='replace')
for key in ['rynn32','bell_rm']:
 p=O/key/'report.json'
 if p.is_file():
  r=json.loads(p.read_text());out[key]={k:r[k] for k in ['engineering_passed','summary','error','elapsed_s','thresholds','strict_load'] if k in r};out[key]['completed']=len(r.get('cases',r.get('rows',[])))
capture={}
labels={}
for key in ['native_rynn32','native_bell32']:
 p=O/key/'capture';paths=list(p.glob('native-*.json')) if p.exists() else [];capture[key]=len(paths)
 rows=[json.loads(path.read_text()) for path in paths]
 labels[key]=dict(success=sum(r['reference_success'] for r in rows),failure=sum(not r['reference_success'] for r in rows))
out['capture_counts']=capture
out['native_labels']=labels
out['lifecycle_receipts']={name:(O/name).is_file() for name in ['cleanup.json','smoke-release.json','rlt-return-dispatched.json','rlt-status.json','rlt-first-round-pending.json']}
r=subprocess.run(['nvidia-smi','pmon','-c','1','-s','um'],text=True,capture_output=True,timeout=15)
ours=[]
for line in r.stdout.splitlines():
 parts=line.split()
 if not parts or parts[0].startswith('#') or len(parts)<3 or not parts[1].isdigit():continue
 p=Path('/proc')/parts[1]
 try:
  if p.stat().st_uid==os.getuid():ours.append(line)
 except FileNotFoundError:pass
out['our_gpu_contexts']=ours
print(json.dumps(out),flush=True)
