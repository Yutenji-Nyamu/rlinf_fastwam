"""Correct only output-route rewriting of the original complete checkpoint path."""
import ast,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-two-gpu-b32-v1';O=S/'runs/lift-two-gpu-b32-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists() and not (D/'ready.json').exists()
path=D/'prepared/owner-plan.json';p=read(path);changes=[]
source=D/'code/prepare.py';assert sha(source)==p['source_sha256'][str(source)]
text=source.read_text();a="    formal=route(formal,'formal')\n";assert text.count(a)==1
text=text.replace(a,a+"    formal['runner']['resume_dir']=str(resume)\n");ast.parse(text)
changes.append({'path':str(source),'before':sha(source)})
source.chmod(0o600);source.write_text(text);source.chmod(0o500);p['source_sha256'][str(source)]=sha(source)
for row in p['trials']:
 f=Path(row['config']);assert sha(f)==p['source_sha256'][str(f)]
 cfg=read(f);assert cfg['runner']['resume_dir'].startswith(str(O/row['key']))
 changes.append({'path':str(f),'before':sha(f),'old_resume':cfg['runner']['resume_dir']})
 cfg['runner']['resume_dir']=p['resume_dir'];f.write_text(json.dumps(cfg,indent=2)+'\n');p['source_sha256'][str(f)]=sha(f)
path.write_text(json.dumps(p,indent=2)+'\n')
(D/'resume-path-repair.json').write_text(json.dumps(changes,indent=2)+'\n')
code='''import importlib.util,json,hashlib,sys
from pathlib import Path
D=Path(%r);path=D/'prepared/owner-plan.json';p=json.loads(path.read_text())
spec=importlib.util.spec_from_file_location('two_validated_owner',D/'code/owner.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
module=m.install(p);module.validate(p)
value={'plan':str(path),'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'entrypoint':str(D/'code/owner.py'),'cpu_validate_passed':True,'resume_step':p['resume_step'],'physical_gpus':[4,5]}
with (D/'ready.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
''' % str(D)
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
r=subprocess.run([py,'-B','-c',code],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=180)
print(json.dumps({'exit_code':r.returncode,'stdout':r.stdout[-3000:],'stderr':r.stderr[-5000:]}))
