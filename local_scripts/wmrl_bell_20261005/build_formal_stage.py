"""Serialize the reviewed small source set; project checks run on SZ3 only."""
import base64
import hashlib
import json
from pathlib import Path

W = Path(__file__).resolve().parents[2]
O = Path(__file__).resolve().parent
files = {}
for folder, sub in [('local_scripts/rynn_binary_20261005', 'binary'), ('local_patches/click_bell_20261005', 'bell')]:
    for p in (W / folder).glob('*.py'):
        raw = p.read_bytes()
        files[sub + '/' + p.name] = dict(data=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
for name in ['prechecks.py', 'test_prechecks.py', 'reuse_scope_hook.py', 'prepare_formal_remote.py']:
    raw = (O / name).read_bytes()
    files['ops/' + name] = dict(data=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
payload = base64.b64encode(json.dumps(files).encode()).decode()
body = r'''
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'click-bell-v1';C=D/'code'
assert not C.exists() and not (D/'prepared').exists()
C.mkdir(mode=0o700)
for rel,row in json.loads(base64.b64decode(PAYLOAD)).items():
    p=C/rel;assert p.resolve().is_relative_to(C.resolve())
    p.parent.mkdir(exist_ok=True);raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];p.write_bytes(raw)
    compile(raw,str(p),'exec')
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
checks=[]
for sub,module in [('binary','test_binary_probe'),('bell','test_click_bell'),('ops','test_prechecks')]:
    r=subprocess.run([py,'-B','-m','unittest',module,'-v'],cwd=C/sub,env=env,text=True,capture_output=True,timeout=180)
    row=dict(module=module,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);checks.append(row);print(json.dumps(row),flush=True)
    if r.returncode:break
(D/'cpu-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
assert len(checks)==3 and all(row['returncode']==0 for row in checks)
r=subprocess.run([py,'-B',str(C/'ops/prepare_formal_remote.py')],env=env,text=True,capture_output=True,timeout=180)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)),flush=True)
assert r.returncode==0
'''
(O / 'formal_stage_remote.py').write_text('PAYLOAD=' + repr(payload) + '\n' + body, encoding='utf-8')
print(json.dumps(dict(files=len(files), rpc=str(O / 'formal_stage_remote.py'))))
