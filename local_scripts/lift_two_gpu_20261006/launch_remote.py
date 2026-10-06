"""Launch the reviewed two-card owner once; no independent job signals."""
import datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-two-gpu-b32-v1';O=S/'runs/lift-two-gpu-b32-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ready=read(D/'ready.json');plan=read(ready['plan'])
assert ready['cpu_validate_passed'] and ready['physical_gpus']==plan['physical_gpus']==[4,5]
assert sha(ready['plan'])==ready['plan_sha256']
assert not (D/'launch.json').exists() and not O.exists()
assert plan['owner_dir']==str(O) and plan['mode']=='two_gpu_b32_formal'
for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
assert subprocess.check_output(['git','-C',plan['repo'],'rev-parse','HEAD'],text=True).strip()==plan['repo_head']
old=read(S/'runs/lift-pot-v1/final.json');assert old['rlt_return_dispatched'] and old['recovery_error'] is None
argv=[plan['python'],'-u','-B',ready['entrypoint'],'--plan',ready['plan'],'owner']
with (D/'owner-console.log').open('x') as f:
 child=subprocess.Popen(argv,cwd=plan['repo'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
p=Path('/proc')/str(child.pid);st=(p/'stat').read_text().split(') ',1)[1].split()
assert p.stat().st_uid==20001 and st[0] not in ['Z','X']
receipt={'time':datetime.datetime.now().astimezone().isoformat(),'pid':child.pid,'uid':20001,'start':int(st[19]),'argv':argv,'plan_sha256':ready['plan_sha256'],'physical_gpus':[4,5],'gpu6_gpu7_rlt_unchanged':True}
with (D/'launch.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
