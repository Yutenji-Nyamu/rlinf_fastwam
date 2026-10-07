"""Publish this reviewed change on the existing WMRL branch; never touch the runtime repo."""
import ast,base64,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent; W=H.parents[1]; D=W/'docs/world-model/task_reward_plan_20261006'
prior=json.loads((D/'two_gpu_b16_started_published.json').read_text())
names=['docs/world-model/task_reward_plan_20261006/'+n for n in [
 'README.md','two_gpu_b16_restart_20261007.md','two_gpu_b16_simplification_20261007.md',
 'two_gpu_b16_monitor_20261007.json','two_gpu_b16_lean_light.json']]
names+=['local_scripts/wm_b16_retry_20261007/'+n for n in [
 'owner.py','prepare.py','test_owner.py','make_package.py','launch_remote.py',
 'status_remote.py','record.py','make_publication.py']]
delete=['local_scripts/wm_b16_retry_20261007/'+n for n in ['build_owner.py','start_audit_remote.py']]
files={}
for name in names:
 raw=(W/name).read_bytes().replace(b'\r\n',b'\n'); assert len(raw)<262144
 if name.endswith('.py'): ast.parse(raw)
 files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
P={'files':files,'delete':delete,'prior':prior['commit'],'branch':prior['branch'],'runtime_head':prior['runtime_head']}
REMOTE=r'''
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
W=S/'publication/wmrl-bell-release-v1'; R=S/'rlinf-opendw-two-gpu-v1'
F=S/'lift-two-gpu-b16-lean-20261007-v1'; E=S/'publication/lift-b16-lean-20261007-v1'
def git(p,*args): return subprocess.check_output(['git','-C',str(p),*args],text=True,timeout=180).strip()
assert git(R,'rev-parse','HEAD')==P['runtime_head'] and git(R,'status','--porcelain')==''
assert git(W,'rev-parse','HEAD')==P['prior'] and git(W,'branch','--show-current')==P['branch']
assert git(W,'status','--porcelain','--untracked-files=all')=='' and not E.exists()
assert git(W,'remote','get-url','personal')=='git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
assert git(W,'ls-remote','--heads','personal','refs/heads/'+P['branch']).split()[0]==P['prior']
for name in ['owner.py','prepare.py','test_owner.py']:
 assert hashlib.sha256((F/'code'/name).read_bytes()).hexdigest()==P['files']['local_scripts/wm_b16_retry_20261007/'+name]['sha256']
E.mkdir(mode=0o700)
for name,row in P['files'].items():
 raw=base64.b64decode(row['base64'],validate=True); assert hashlib.sha256(raw).hexdigest()==row['sha256']
 path=W/name; assert path.resolve().is_relative_to(W.resolve()) and not path.is_symlink()
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for name in P['delete']:
 path=W/name; assert path.resolve().is_relative_to(W.resolve()) and not path.is_symlink()
 if path.exists(): path.unlink()
git(W,'add','-A','--',*P['files'],*P['delete'])
changed=git(W,'diff','--cached','--name-only').splitlines()
assert changed and set(changed)<=set(P['files'])|set(P['delete'])
git(W,'diff','--cached','--check')
for name in changed:
 if name in P['files']:
  raw=subprocess.check_output(['git','-C',str(W),'show',':'+name])
  assert hashlib.sha256(raw).hexdigest()==P['files'][name]['sha256']
git(W,'commit','-m','Simplify WMRL supervision and restart unchanged B16 formal training')
head=git(W,'rev-parse','HEAD');assert git(W,'rev-parse','HEAD^')==P['prior']
git(W,'push','personal','HEAD:refs/heads/'+P['branch'])
remote=git(W,'ls-remote','--heads','personal','refs/heads/'+P['branch']).split()[0];assert remote==head
assert git(R,'rev-parse','HEAD')==P['runtime_head'] and git(R,'status','--porcelain')==''
receipt={'time':datetime.datetime.now().astimezone().isoformat(),'branch':P['branch'],'commit':head,
 'verified_remote_commit':remote,'files':changed,'pushed':True,'runtime_head':P['runtime_head'],'runtime_unchanged':True}
(E/'published.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
source='P = '+repr(P)+'\n'+REMOTE;ast.parse(source)
(H/'publication_remote.py').write_text(source,encoding='utf-8',newline='\n')
print(json.dumps({'files':len(files),'deletions':len(delete),'prior':P['prior']}))
