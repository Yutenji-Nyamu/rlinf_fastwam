"""Stage final task-only owner bindings; no training is started here."""
import ast,base64,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE=r'''
import base64,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-pot-v1';C=D/'code'
assert not (D/'prepared/owner-plan.json').exists() and not (S/'runs/lift-pot-v1').exists()
prior={'rm_adapter.py':'155cbc5b38ae9d4c90948a064902679c801078e47a87b2c3f836112fbb90131b','patch_lift.py':'94723459bdc73c26b407257ccdde94a3d756081256b8ca0bd6516f87339dde47'}
for name,row in PAYLOAD.items():
 raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 p=C/name
 if p.exists():
  assert name in prior and hashlib.sha256(p.read_bytes()).hexdigest() in [prior[name],row['sha256']]
  p.chmod(0o600);p.write_bytes(raw)
 else:p.write_bytes(raw)
 p.chmod(0o500)
cmd=['/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python','-B',str(C/'assemble_owner.py'),'--recipe-dir',str(D/'prepared'),'--private-repo',str(S/'rlinf-opendw-bell-v1')]
result=subprocess.run(cmd,cwd=C,env=dict(os.environ,PYTHONPATH=str(C)),text=True,capture_output=True,timeout=300)
print(json.dumps({'rc':result.returncode,'stdout':result.stdout[-10000:],'stderr':result.stderr[-5000:]}),flush=True)
assert result.returncode==0
'''
def main():
 payload={}
 for name in ['rm_adapter.py','patch_lift.py','assemble_owner.py','native_prechecks.py']:
  raw=(HERE/name).read_bytes();ast.parse(raw);payload[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 source='PAYLOAD='+repr(payload)+'\n'+REMOTE;ast.parse(source)
 (HERE/'assemble_remote.py').write_text(source,encoding='utf-8')
 print(json.dumps({'files':list(payload),'launch':False}))
if __name__=='__main__':main()
