import base64,hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent
files={p.name:base64.b64encode(p.read_bytes()).decode() for p in [O/'prepare_cycles.py',O/'test_prepare_cycles.py']}
body=r'''
import base64,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-binary-v1');C=D/'cycle-code-v2';assert not C.exists();C.mkdir()
for name,data in json.loads(base64.b64decode(PAYLOAD)).items():(C/name).write_bytes(base64.b64decode(data))
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
commands=[[py,'-B','-m','unittest','test_prepare_cycles','-v'],[py,'-u','-B',str(C/'prepare_cycles.py'),'--prepared-dir',str(D/'prepared-cycles'),'--cycle-prefix','bell-v1','--groups','all']]
for i,cmd in enumerate(commands):
    r=subprocess.run(cmd,cwd=C,env=env,capture_output=True,text=True,timeout=240)
    row=dict(argv=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);(D/f'cycle-prepare-{i}.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
    if r.returncode:raise RuntimeError('Cycle preparation failed; nothing stopped')
'''
(O/'cycles_remote.py').write_text('PAYLOAD='+repr(base64.b64encode(json.dumps(files).encode()).decode())+'\n'+body,encoding='utf-8')
