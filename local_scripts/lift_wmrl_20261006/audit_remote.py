"""Read-only source/cloud/resource refresh for the exact lift owner."""
import datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-pot-v1';O=S/'runs/lift-pot-v1'
read=lambda p:json.loads(p.read_text())
plan=read(D/'prepared/owner-plan.json')
out={'time':datetime.datetime.now().astimezone().isoformat(),'source_changed':[],'receipts':{}}
for path,expected in plan['source_sha256'].items():
 p=Path(path)
 if p.suffix in ['.py','.yaml'] and hashlib.sha256(p.read_bytes()).hexdigest()!=expected:out['source_changed'].append(path)
for rel in ['error.json','recovery-error.json','native_lift32/driver-identity.json','native_lift32/ray-job.json','native_lift32/summary.json','startup_smoke/result.json','formal/driver-identity.json','final.json']:
 p=O/rel
 if p.exists():
  value=read(p)
  if rel.endswith('ray-job.json'):value={k:value[k] for k in ['driver_pid','namespace','job_id'] if k in value}
  out['receipts'][rel]=value
out['resources']={}
for k in ['MemAvailable','MemTotal']:
 out['resources'][k]=next(x.strip() for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(k+':'))
out['resources']['loadavg']=Path('/proc/loadavg').read_text().strip()
out['resources']['disk']=subprocess.check_output(['df','-h','/','/data/chenyiteng'],text=True,timeout=10)
out['git_local_head']=subprocess.check_output(['git','-C',str(S/'publication/wmrl-bell-release-v1'),'rev-parse','HEAD'],text=True,timeout=10).strip()
cmd=['git','-C',str(S/'publication/wmrl-bell-release-v1'),'ls-remote','--heads','personal','refs/heads/codex/wmrl-bell-reward-20261005']
p=subprocess.run(cmd,text=True,capture_output=True,timeout=90);out['git_remote']={'rc':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-1000:]}
print(json.dumps(out))
