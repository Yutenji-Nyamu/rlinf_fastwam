"""Serialize exact small sources for reviewed remote staging; no local tests."""
import base64,hashlib,json
from pathlib import Path
W=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
files={}
for folder,sub in [('local_scripts/rynn_binary_20261005','binary'),('local_patches/click_bell_20261005','bell')]:
    for p in (W/folder).glob('*.py'):
        raw=p.read_bytes();files[sub+'/'+p.name]={'data':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
p=O/'prepare_cycles.py'
if p.exists():
    raw=p.read_bytes();files['prepare_cycles.py']={'data':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
payload=base64.b64encode(json.dumps(files).encode()).decode()
body=r'''
import base64,datetime,hashlib,json,os,shutil,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'rynn-binary-v1';R=S/'rlinf-rynn-binary-v1'
assert not D.exists() and not R.exists()
D.mkdir(mode=0o700);C=D/'code';C.mkdir()
for rel,row in json.loads(base64.b64decode(PAYLOAD)).items():
    p=C/rel;assert p.resolve().is_relative_to(C.resolve())
    p.parent.mkdir(exist_ok=True);raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];p.write_bytes(raw)
    compile(raw,str(p),'exec')
base=S/'rlinf-multigpu-v1';head=subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip();assert head=='2151a08ee1bd75df1bef0d8190e594bd5c7f7977'
subprocess.run(['git','-C',str(base),'worktree','add','-b','codex/rynn-binary-20261005',str(R),head],check=True)
for rel in ['rlinf/envs/__init__.py','rlinf/runners/embodied_runner.py','rlinf/envs/world_model/opendw_adapter.py','rlinf/envs/world_model/opendw_robotwin_env.py']:
    p=R/rel;p.parent.mkdir(exist_ok=True);shutil.copyfile(base/rel,p)
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
scope=S/'rynn-numeric-v1/rlt-return-repair-v2/scope/environment-fragment.json'
env=json.loads((S/'formal-b16-control-v1/prepared/environment.json').read_text());env.update(json.loads(scope.read_text()))
env={k:str(v).replace(str(base),str(R)) for k,v in env.items() if k!='CUDA_VISIBLE_DEVICES'}
env['PYTHONPATH']=':'.join(dict.fromkeys([env['PYTHONPATH'].split(':')[0],str(R),*env['PYTHONPATH'].split(':')[1:]]))
env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='1';env['RYNN_BINARY_CAPTURE_DIR']=str(D/'capture')
(D/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
commands=[
 [py,'-B','-m','unittest','test_binary_probe','-v'],
 [py,'-B',str(C/'binary/install_capture.py'),'--private-env-file',str(R/'rlinf/envs/robotwin/robotwin_env.py')],
 [py,'-B',str(C/'binary/prepare_native_eval.py'),'--formal-config',str(S/'formal-b16-control-v1/prepared/formal.yaml'),'--checkpoint-file',str(S/'runs/formal-b16-v1/formal/run/opendw-adjust-bottle-formal-b16-v1/checkpoints/global_step_70/actor/model_state_dict/full_weights.pt'),'--output-dir',str(D/'native')]]
checks=[]
for argv in commands:
    r=subprocess.run(argv,cwd=C/'binary',env=cpu,text=True,capture_output=True,timeout=180)
    checks.append({'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});print(json.dumps(checks[-1]),flush=True)
    if r.returncode:break
out={'time':datetime.datetime.now().astimezone().isoformat(),'checks':checks,'prepared':len(checks)==len(commands) and all(r['returncode']==0 for r in checks),'repo':str(R),'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.rglob('*.py')}}
(D/'prepared.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True);assert out['prepared']
'''
(O/'stage_remote.py').write_text('PAYLOAD='+repr(payload)+'\n'+body,encoding='utf-8')
print(json.dumps({'files':len(files),'script':str(O/'stage_remote.py')}))
