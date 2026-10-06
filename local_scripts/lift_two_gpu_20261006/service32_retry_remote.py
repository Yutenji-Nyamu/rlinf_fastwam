"""Prepare and launch attempt 2 after the CPU-only CLI rejection of B32."""
import ast,copy,datetime,hashlib,importlib.util,json,os,socket,subprocess,sys
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-two-gpu-b32-v1'
O1=S/'runs/lift-two-gpu-b32-v1';O=S/'runs/lift-two-gpu-b32-v2'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
final=read(O1/'final.json');assert not final['rlt_borrowed'] and final['recovery_error'] is None
assert final['error']['error']=='Service exited before CPU readiness' and read(O1/'cleanup.json')['all_stopped']
old=read(O1/'owner-identity.json');proc=Path('/proc')/str(old['pid'])
if proc.exists():assert int((proc/'stat').read_text().split(') ',1)[1].split()[19])!=old['start']
assert not O.exists() and not (D/'launch-v2.json').exists()
p=read(D/'prepared/owner-plan.json')
for name,digest in p['source_sha256'].items():assert sha(name)==digest,name
target=D/'generated/service-v2';target.mkdir(mode=0o700)
for source in (D/'generated/service').glob('*.py'):
 text=source.read_text()
 if source.name=='opendw_service_batched.py':
  a='choices=(1, 16), default=16';assert text.count(a)==1;text=text.replace(a,'choices=(1, 16, 32), default=16')
  text=text.replace('Use batched/16 or explicit b1_reference/1','Use batched/16, batched/32 or explicit b1_reference/1')
 ast.parse(text);dest=target/source.name;dest.write_text(text);dest.chmod(0o500)
 p['source_sha256'][str(dest)]=sha(dest)
prepared=D/'prepared-v2';prepared.mkdir(mode=0o700)
for row in p['trials']:
 source=Path(row['config']);cfg=read(source)
 cfg=json.loads(json.dumps(cfg).replace(str(O1),str(O)))
 assert cfg['runner']['resume_dir']==p['resume_dir']
 for c in ['actor','env','rollout']:cfg[c]['group_name']+='v2'
 dest=prepared/source.name;dest.write_text(json.dumps(cfg,indent=2)+'\n');dest.chmod(0o600)
 row['config']=str(dest);row['namespace']+='v2';p['source_sha256'][str(dest)]=sha(dest)
service=p['services'][0];service['cwd']=str(target)
service['argv']=[s.replace(str(D/'generated/service'),str(target)).replace(str(O1),str(O)) for s in service['argv']]
service['environment']={k:v.replace(str(D/'generated/service'),str(target)) for k,v in service['environment'].items()}
p['owner_dir']=str(O);p['retry_of']=str(O1)
path=prepared/'owner-plan.json';path.write_text(json.dumps(p,indent=2)+'\n');path.chmod(0o600)
spec=importlib.util.spec_from_file_location('retry_owner',D/'code/owner.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
module=m.install(p);module.validate(p)
ready={'plan':str(path),'plan_sha256':sha(path),'entrypoint':str(D/'code/owner.py'),'cpu_validate_passed':True,'physical_gpus':[4,5],'resume_step':p['resume_step']}
(D/'ready-v2.json').write_text(json.dumps(ready,indent=2)+'\n')
argv=[p['python'],'-u','-B',ready['entrypoint'],'--plan',str(path),'owner']
with (D/'owner-v2-console.log').open('x') as f:
 child=subprocess.Popen(argv,cwd=p['repo'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
proc=Path('/proc')/str(child.pid);st=(proc/'stat').read_text().split(') ',1)[1].split()
assert proc.stat().st_uid==20001
receipt={'time':datetime.datetime.now().astimezone().isoformat(),'pid':child.pid,'uid':20001,'start':int(st[19]),'argv':argv,'plan_sha256':sha(path),'physical_gpus':[4,5],'retry_reason':'CLI whitelist adds B32; prior attempt borrowed no GPUs'}
(D/'launch-v2.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
