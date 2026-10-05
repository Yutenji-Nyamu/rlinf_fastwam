"""Serialize only the retry preparers and repaired native entrypoint."""
import base64,json
from pathlib import Path
O=Path(__file__).resolve().parent;W=O.parents[1]
files={'ops/prepare_formal_v2.py':O/'prepare_formal_v2.py','ops/prepare_cycles_v2.py':O/'prepare_cycles_v2.py','binary/run_native_eval.py':W/'local_scripts/rynn_binary_20261005/run_native_eval.py'}
payload={name:base64.b64encode(path.read_bytes()).decode() for name,path in files.items()}
body=r'''
import base64,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/click-bell-v2');assert not D.exists()
for name,data in PAYLOAD.items():
 p=D/'code'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
r=subprocess.run([py,'-B',str(D/'code/ops/prepare_formal_v2.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),text=True,capture_output=True,timeout=180)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)),flush=True)
assert r.returncode==0
'''
(O/'stage_v2_remote.py').write_text('PAYLOAD='+repr(payload)+'\n'+body,encoding='utf-8')
