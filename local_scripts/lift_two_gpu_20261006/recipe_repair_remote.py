"""Repair the prelaunch scope-id lookup; resume an otherwise empty recipe stage."""
import hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-two-gpu-b32-v1'
assert not (D/'ready.json').exists() and not (S/'runs/lift-two-gpu-b32-v1').exists()
assert not list((D/'cycles').iterdir())
p=D/'code/prepare.py';text=p.read_text();before=hashlib.sha256(p.read_bytes()).hexdigest()
for a,b in [("cycles=D/'cycles';cycles.mkdir(mode=0o700)","cycles=D/'cycles';cycles.mkdir(mode=0o700,exist_ok=True)"),("scope_id=read(manifest)['scope_id']","scope_id=prior_plan['scope_id']")]:
 assert text.count(a)==1;text=text.replace(a,b)
p.chmod(0o600);p.write_text(text);p.chmod(0o500)
record={'file':str(p),'before':before,'after':hashlib.sha256(p.read_bytes()).hexdigest(),'reason':'scope_id belongs to the verified prior lifecycle plan'}
(D/'scope-lookup-repair.json').write_text(json.dumps(record,indent=2))
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
with (D/'recipe-preparation-retry.log').open('x') as f:
 r=subprocess.run([py,'-B',str(p),'recipe'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,timeout=300)
print(json.dumps({'exit_code':r.returncode,'log_tail':(D/'recipe-preparation-retry.log').read_text()[-6000:]}))
if r.returncode==0:print((D/'ready.json').read_text())
