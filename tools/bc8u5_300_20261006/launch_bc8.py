import ast,hashlib,json,os,subprocess,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-bc8u5-300-v1');p=json.loads((S/'bc/plan.json').read_text());script=S/'cutover_owner_v2.py'
ast.parse(script.read_text());assert not (S/'cutover-launch-v2.json').exists()
assert json.loads((S/'cpu-tests.json').read_text())['rc']==0
env={k:v for k,v in os.environ.items() if k not in ('CUDA_VISIBLE_DEVICES','LD_PRELOAD','PYTHONPATH','RLINF_OPENDW_GPU_SCOPE_MANIFEST')};env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
with (S/'cutover-v2.log').open('x') as f:q=subprocess.Popen([p['python'],'-u','-B',str(script)],cwd=p['repo'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
v={'time':time.time(),'pid':q.pid,'script':str(script),'sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'user_authorized':'GPU4 only; fresh N8/U5/300; fixed instruction and <=3 chunks; preserve old outputs; others unchanged'}
(S/'cutover-launch-v2.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
