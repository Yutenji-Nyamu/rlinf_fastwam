"""Read current RLT45 ownership and source pins before a batch-only restart."""
import json,os,socket,subprocess,hashlib
from pathlib import Path
from datetime import datetime
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
F=S/'rlt45-return-20261007-v1';P=S/'lift-two-gpu-from0-v2/prepared/owner-plan.json'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(P);cp=read(F/'cycle/plan.json');out={'time':datetime.now().astimezone().isoformat(),'rlt':{},'files':{},'sources':{}}
for g in [4,5]:
 sub=read(F/f'return-g{g}/plan.json');r=sub['runs'][f'gpu{g}'];rt=Path(r['new_run'])/'runtime';i=read(rt/'driver-identity.json')
 proc=Path('/proc')/str(i['pid']);st=(proc/'stat').read_text().rsplit(')',1)[1].split()
 assert int(st[19])==i['start'] and proc.stat().st_uid==20001
 out['rlt'][str(g)]={'identity':i,'run':r['new_run'],'tail':(rt/'driver.log').read_text(errors='replace')[-2600:]}
for p in [F/'cycle/rlt_returned_multigpu_cycle.py',Path(plan['base_owner_module'])]:out['sources'][str(p)]=p.read_text()
out['plan_fields']={k:v for k,v in plan.items() if k not in ['environment_file','source_sha256']}
out['source_count']=len(plan['source_sha256']);out['source_pins_match']=all(sha(p)==h for p,h in plan['source_sha256'].items())
for p in [F/'audit.json',F/'dispatched.json',F/'cycle/resumed-dispatched.json',F/'scope/scope.json',F/'scope/activation.json']:out['files'][str(p)]=read(p)
out['profiles']={str(p):{'sha256':sha(p),'json':read(p)} for p in Path('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d').glob('*.json')}
out['gpu']=subprocess.check_output(['nvidia-smi'],text=True,timeout=20)
out['runtime_processes']=subprocess.check_output(['ps','-u','chenyiteng','-o','pid,ppid,pcpu,args','--no-headers'],text=True,timeout=10)
out['runtime_processes']='\n'.join(x for x in out['runtime_processes'].splitlines() if any(t in x for t in ['owner.py','fresh_owner.py','rlt_returned_cycle.py','opendw_service_batched.py']) and 'stdin' not in x)
print(json.dumps(out))
