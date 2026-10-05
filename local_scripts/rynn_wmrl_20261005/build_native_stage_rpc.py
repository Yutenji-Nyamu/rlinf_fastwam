"""Serialize the last native-input diagnostic; server executes checks."""
import base64,hashlib,json
from pathlib import Path
ops=Path(__file__).resolve().parent
names=['rynn_native_diagnostic.py','rynn_diagnostic_owner_v2.py','test_rynn_diagnostic_owner_v2.py',
       'rynn_diagnostic_owner.py','test_rynn_diagnostic_owner.py']
files={n:{'base64':base64.b64encode((ops/n).read_bytes()).decode(),
    'sha256':hashlib.sha256((ops/n).read_bytes()).hexdigest()} for n in names}
remote=r'''
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'rynn-diagnosis-v2';C=D/'code'
assert D.is_dir() and D.stat().st_uid==20001 and not (D/'prepared/plan.json').exists() and not (D/'run').exists()
archive=D/'failed-cpu-preflight-v1';assert not archive.exists();archive.mkdir(mode=0o700)
for p in C.glob('*.py'):(archive/p.name).write_bytes(p.read_bytes())
(archive/'cpu-tests.json').write_bytes((D/'prepared/cpu-tests.json').read_bytes())
for name,row in json.loads(base64.b64decode(PAYLOAD)).items():
    assert Path(name).name==name
    raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
    (C/name).write_bytes(raw);compile(raw,str(C/name),'exec')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
py='/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python'
preflight=[py,'-B',str(C/'rynn_native_diagnostic.py'),'--cpu-preflight',
    '--service-module',str(S/'rynn-control-v2/code/rynn_success_service.py'),
    '--model-path','/data/chenyiteng/models/RynnValue-8B-8738c5e4',
    '--manifest-path','/data/chenyiteng/models/RynnValue-8B-8738c5e4/manifest.json',
    '--physical-gpu','4','--cases-json',str(S/'rynn-diagnosis-v1/prepared/cases.json'),
    '--controls-json',str(S/'rynn-control-v1/prepared/rm-sanity-clips.json'),
    '--official-inference','/data/chenyiteng/projects/RynnValue-10e0d333/rynn_infer/inference.py',
    '--native-frames-npz',str(D/'prepared/native_frames.npz'),
    '--native-frames-json',str(D/'prepared/native_frames.json'),
    '--samples-npz',str(S/'rynn-diagnosis-v1/prepared/samples.npz'),
    '--output',str(D/'preflight-v2/result.json')]
checks=[]
for command in [[py,'-B','-m','unittest','test_rynn_diagnostic_owner_v2','-v'],preflight]:
    result=subprocess.run(command,cwd=C,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    checks.append(dict(command=command,exit_code=result.returncode,output=result.stdout))
receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),checks=checks,
    all_cpu_tests_passed=all(row['exit_code']==0 for row in checks),
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.glob('*.py')})
(D/'prepared/cpu-tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));assert receipt['all_cpu_tests_passed']
'''
target=ops/'repair_native_remote.py'
target.write_text('PAYLOAD = '+repr(base64.b64encode(json.dumps(files).encode()).decode())+'\n'+remote,encoding='utf-8')
print(json.dumps({'path':str(target),'files':len(files),'bytes':target.stat().st_size}))
