"""Serialize reviewed files; run all project checks on SZ3 with CUDA hidden."""
import base64,hashlib,json
from pathlib import Path
ops=Path(__file__).resolve().parent
sources=[ops/n for n in ('rynn_numeric_probe.py','rynn_numeric_owner.py','test_numeric_probe.py','test_numeric_owner.py','prepare_dataset.py')]
sources += [ops.parent/'rynn_wmrl_20261005'/n for n in ('rynn_diagnostic_owner.py','test_rynn_diagnostic_owner.py','rynn_diagnostic_owner_v2.py','test_rynn_diagnostic_owner_v2.py')]
files={p.name:dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources}
remote=r'''
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'rynn-numeric-v1';C=D/'code'
assert not D.exists();D.mkdir(mode=0o700);C.mkdir();(D/'prepared').mkdir();(D/'preflight').mkdir()
for name,row in json.loads(base64.b64decode(PAYLOAD)).items():
    assert Path(name).name==name
    raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
    (C/name).write_bytes(raw);compile(raw,str(C/name),'exec')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
py='/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python'
rlpy='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
preflight=[py,'-B',str(C/'rynn_numeric_probe.py'),'--cpu-preflight',
    '--service-module',str(S/'rynn-control-v2/code/rynn_success_service.py'),
    '--model-path','/data/chenyiteng/models/RynnValue-8B-8738c5e4',
    '--manifest-path','/data/chenyiteng/models/RynnValue-8B-8738c5e4/manifest.json',
    '--physical-gpu','4','--cases-json',str(D/'prepared/cases.json'),
    '--official-inference','/data/chenyiteng/projects/RynnValue-10e0d333/rynn_infer/inference.py',
    '--samples-npz',str(D/'prepared/samples.npz'),'--output',str(D/'preflight/result.json')]
checks=[]
for command in [[py,'-B','-m','unittest','test_numeric_owner','test_numeric_probe','-v'],[rlpy,'-B',str(C/'prepare_dataset.py')],preflight]:
    result=subprocess.run(command,cwd=C,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
    row=dict(command=command,exit_code=result.returncode,output=result.stdout);checks.append(row)
    print(json.dumps(row),flush=True)
    if result.returncode:break
receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),checks=checks,
    all_cpu_tests_passed=len(checks)==3 and all(row['exit_code']==0 for row in checks),
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.glob('*.py')})
(D/'prepared/cpu-tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));assert receipt['all_cpu_tests_passed']
'''
target=ops/'stage_remote.py'
target.write_text('PAYLOAD = '+repr(base64.b64encode(json.dumps(files).encode()).decode())+'\n'+remote,encoding='utf-8')
print(json.dumps(dict(path=str(target),files=len(files),bytes=target.stat().st_size)))
