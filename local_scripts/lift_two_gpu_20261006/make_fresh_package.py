"""Package only the fresh-start owner and its CPU regression check."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = {}
for name in ['fresh_owner.py', 'prepare_fresh.py', 'test_resource_timeout.py']:
    raw = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
    ast.parse(raw, filename=name)
    files[name] = {'base64': base64.b64encode(raw).decode(), 'sha256': hashlib.sha256(raw).hexdigest()}
source = 'FILES = ' + repr(files) + '\n' + r'''
import ast,base64,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
F=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-two-gpu-from0-v2')
assert not F.exists()
F.mkdir(mode=0o700);(F/'code').mkdir(mode=0o700)
for name,row in FILES.items():
 assert '/' not in name and name.endswith('.py')
 raw=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 ast.parse(raw);path=F/'code'/name;path.write_bytes(raw);path.chmod(0o500)
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
with (F/'preparation.log').open('x') as stream:
 result=subprocess.run([py,'-B',str(F/'code/prepare_fresh.py')],
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1'),
  stdout=stream,stderr=subprocess.STDOUT,timeout=300)
print(json.dumps({'exit_code':result.returncode,'tail':(F/'preparation.log').read_text()[-8000:]}),flush=True)
assert result.returncode==0
'''
ast.parse(source)
(HERE / 'stage_fresh_remote.py').write_text(source, encoding='utf-8', newline='\n')
print(json.dumps({'files': list(files), 'generated_only': True}))
