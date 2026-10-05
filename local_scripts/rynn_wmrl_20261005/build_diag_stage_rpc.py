"""Package diagnosis-only files; all project checks execute on SZ3."""
import base64,hashlib,json
from pathlib import Path
ops=Path(__file__).resolve().parent
names=['rynn_diagnostic.py','rynn_diagnostic_owner.py','test_rynn_diagnostic_owner.py','preflight_rynn_diagnostic.py']
files={name:{'base64':base64.b64encode((ops/name).read_bytes()).decode(),
    'sha256':hashlib.sha256((ops/name).read_bytes()).hexdigest()} for name in names}
remote=r'''
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-diagnosis-v1');C=D/'code'
assert not C.exists() and not (D/'prepared/plan.json').exists()
C.mkdir(mode=0o700)
for name,row in json.loads(base64.b64decode(PAYLOAD)).items():
    assert Path(name).name==name
    raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
    (C/name).write_bytes(raw);compile(raw,str(C/name),'exec')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
py='/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python'
checks=[]
preflight=[py,'-B',str(C/'preflight_rynn_diagnostic.py'),
    '--service-module',str(D.parent/'rynn-control-v2/code/rynn_success_service.py'),
    '--model-path','/data/chenyiteng/models/RynnValue-8B-8738c5e4',
    '--manifest-path','/data/chenyiteng/models/RynnValue-8B-8738c5e4/manifest.json',
    '--physical-gpu','4','--cases-json',str(D/'prepared/cases.json'),
    '--samples-npz',str(D/'prepared/samples.npz'),'--output',str(D/'preflight/result.json')]
for command in [[py,'-B','-m','unittest','test_rynn_diagnostic_owner','-v'],preflight]:
    result=subprocess.run(command,cwd=C,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    checks.append(dict(command=command,exit_code=result.returncode,output=result.stdout))
receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),checks=checks,
    all_cpu_tests_passed=all(row['exit_code']==0 for row in checks),
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.glob('*.py')})
(D/'prepared/cpu-tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));assert receipt['all_cpu_tests_passed']
'''
target=ops/'stage_diag_remote.py'
target.write_text('PAYLOAD = '+repr(base64.b64encode(json.dumps(files).encode()).decode())+'\n'+remote,encoding='utf-8')
print(json.dumps({'path':str(target),'files':len(files),'bytes':target.stat().st_size}))
