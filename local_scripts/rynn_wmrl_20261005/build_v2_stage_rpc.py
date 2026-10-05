"""Serialize the minimal repair packet; tests run only on the server."""
import base64,hashlib,json
from pathlib import Path
W=Path(__file__).resolve().parents[2]
ops=W/'local_scripts/rynn_wmrl_20261005'
patch=W/'local_patches/rynn_wmrl_20261005/tools'
paths=[patch/'rynn_success_service.py',patch/'test_rynn_success_service.py',ops/'rynn_gate_client.py',
       ops/'rynn_formal_owner_v2.py',ops/'prepare_v2_remote.py',ops/'launch_v2_remote.py',ops/'test_rynn_owner_v2.py']
files={p.name:{'base64':base64.b64encode(p.read_bytes()).decode(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
remote=r'''
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D1=S/'rynn-control-v1';D=S/'rynn-control-v2';C=D/'code';R=S/'rlinf-rynn-v1'
assert not D.exists()
old=json.loads((S/'runs/rynn-success-v1/owner-plan.json').read_text())
final=json.loads((S/'runs/rynn-success-v1/final.json').read_text())
assert final['recovery_error'] is None and final['rlt_return_dispatched'] is True
for path,expected in old['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
C.mkdir(parents=True,mode=0o700);(D/'prepared').mkdir(mode=0o700)
for p in (D1/'code').glob('*.py'):
    (C/p.name).write_bytes(p.read_bytes())
for name,row in json.loads(base64.b64decode(PAYLOAD)).items():
    assert Path(name).name==name
    raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
    (C/name).write_bytes(raw)
for p in C.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R)+':'+str(C),OMP_NUM_THREADS='1')
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
commands=[[py,'-B',str(C/'test_env_and_mask.py')],
    ['/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python','-B','-m','unittest','test_rynn_success_service','-v'],
    [py,'-B','-m','unittest','test_rynn_owner_v2','-v']]
checks=[]
for command in commands:
    result=subprocess.run(command,cwd=C,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    checks.append(dict(command=command,exit_code=result.returncode,output=result.stdout))
receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),checks=checks,
    all_cpu_tests_passed=all(row['exit_code']==0 for row in checks),
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.glob('*.py')})
(D/'prepared/cpu-tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));assert receipt['all_cpu_tests_passed']
'''
target=ops/'stage_v2_remote.py'
target.write_text('PAYLOAD = '+repr(base64.b64encode(json.dumps(files).encode()).decode())+'\n'+remote,encoding='utf-8')
print(json.dumps({'path':str(target),'files':len(files),'bytes':target.stat().st_size}))
