"""Package only the consolidated owner and preparation entry for SZ3."""
import ast,base64,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
files={}
for n in ['owner.py','prepare.py','test_owner.py']:
 b=(H/n).read_bytes();ast.parse(b);files[n]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
remote='FILES = '+repr(files)+'\n'+r'''
import ast,base64,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
F=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-two-gpu-b16-lean-20261007-v1')
assert not F.exists();(F/'code').mkdir(parents=True,mode=0o700)
for n,r in FILES.items():
 raw=base64.b64decode(r['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==r['sha256'];ast.parse(raw)
 p=F/'code'/n;p.write_bytes(raw);p.chmod(0o500)
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
test=subprocess.run([py,'-B',str(F/'code/test_owner.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60)
print(json.dumps({'cpu_tests_rc':test.returncode,'stdout':test.stdout,'stderr':test.stderr}),flush=True)
assert test.returncode==0
with (F/'preparation.log').open('x') as log:
 result=subprocess.run([py,'-B',str(F/'code/prepare.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8'),stdout=log,stderr=subprocess.STDOUT,timeout=300)
print(json.dumps({'exit_code':result.returncode,'tail':(F/'preparation.log').read_text()[-5000:]}),flush=True)
assert result.returncode==0
'''
ast.parse(remote);(H/'stage_remote.py').write_text(remote,encoding='utf-8',newline='\n')
