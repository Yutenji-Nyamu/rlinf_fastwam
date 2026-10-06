"""Generate one bounded publication RPC from an explicit reviewed allowlist."""
import ast,base64,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
CODE='local_scripts/lift_two_gpu_20261006/';DOC='docs/world-model/task_reward_plan_20261006/'
FILES=[CODE+n for n in ['owner.py','prepare.py','resume_two_rank.py','test_resume_two_rank.py','batch_probe.py','launch_remote.py','recipe_repair_remote.py','resume_path_repair_remote.py','service32_retry_remote.py','status_v2_remote.py','build_light.py','make_publication.py']]
FILES += [DOC+'two_gpu_b32_execution.md',DOC+'two_gpu_b32_light.json',DOC+'README.md']
def main(resume=False):
 files={}
 for name in FILES:
  raw=(ROOT/name).read_bytes();assert 0<len(raw)<262144
  if name.endswith('.py'):ast.parse(raw)
  files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 payload={'files':files,'prior':'c9aec78686f6329ff76b2c43c8b375ba1dc95848','runtime_head':'9ce50c602c5e773e87e3306218fdde0cf167e8a8','branch':'codex/wmrl-bell-reward-20261005','resume_own_staging':resume}
 source='P = '+repr(payload)+'\n'+REMOTE
 ast.parse(source);(HERE/'publication_remote.py').write_text(source,encoding='utf-8',newline='\n')
 print(json.dumps({'files':len(files),'rpc':'publication_remote.py','generated_only':True}))
REMOTE=r'''
import ast,base64,datetime,hashlib,json,os,re,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');R=S/'rlinf-opendw-two-gpu-v1';D=S/'lift-two-gpu-b32-v1'
W=S/'publication/wmrl-bell-release-v1';E=S/'publication/lift-two-gpu-b32-20261006-v1'
REMOTE='git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
def git(p,*a):return subprocess.check_output(['git','-C',str(p),*a],text=True,timeout=180).strip()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def snapshot():
 assert git(R,'rev-parse','HEAD')==P['runtime_head']
 return {'head':P['runtime_head'],'status':git(R,'status','--porcelain','--untracked-files=all'),'diff_sha256':sha(git(R,'diff','HEAD','--').encode())}
def scan(name,raw):
 assert 0<len(raw)<262144 and Path(name).suffix in ['.py','.md','.json']
 text=raw.decode('utf-8-sig')
 for pattern in [r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'\bsk-[A-Za-z0-9_-]{25,}',r'\bhf_[A-Za-z0-9]{25,}',r'\bAKIA[A-Z0-9]{16}\b']:
  assert not re.search(pattern,text),'Secret-like data in '+name
 if name.endswith('.py'):ast.parse(text)
 if name.endswith('.json'):json.loads(text)
before=snapshot();assert before['status']==''
assert W.stat().st_uid==R.stat().st_uid==20001
assert git(W,'remote','get-url','personal')==REMOTE
assert git(W,'branch','--show-current')==P['branch'] and git(W,'rev-parse','HEAD')==P['prior']
if P['resume_own_staging']:
 assert E.is_dir() and not (E/'published.json').exists() and git(W,'diff','--name-only')==''
 prior_manifest=json.loads((W/'docs/world-model/task_reward_plan_20261006/two_gpu_b32_manifest.json').read_text())
 assert prior_manifest['parent']==P['prior'] and prior_manifest['runtime_head']==P['runtime_head']
 allowed=set(prior_manifest['files'])|{'docs/world-model/task_reward_plan_20261006/two_gpu_b32_manifest.json'}
 assert set(git(W,'diff','--cached','--name-only').splitlines())<=allowed
 assert git(W,'ls-files','--others','--exclude-standard')==''
 for name,digest in prior_manifest['files'].items():assert sha((W/name).read_bytes())==digest,name
else:
 assert git(W,'status','--porcelain','--untracked-files=all')==''
 assert not E.exists();E.mkdir(mode=0o700)
assert git(W,'ls-remote','--heads','personal','refs/heads/'+P['branch']).split()==[P['prior'],'refs/heads/'+P['branch']]
files={}
for name,row in P['files'].items():
 path=Path(name);assert not path.is_absolute() and '..' not in path.parts
 raw=base64.b64decode(row['base64'],validate=True);assert sha(raw)==row['sha256'];scan(name,raw);files[name]=raw
for name in git(R,'diff','HEAD^','HEAD','--name-only').splitlines():
 assert name.startswith('rlinf/') and name.endswith('.py')
 raw=(R/name).read_bytes();scan(name,raw);assert name not in files;files[name]=raw
for name in ['owner.py','prepare.py','resume_two_rank.py','test_resume_two_rank.py','batch_probe.py']:
 assert (D/'code'/name).read_bytes()==files['local_scripts/lift_two_gpu_20261006/'+name],name
manifest={'time':datetime.datetime.now().astimezone().isoformat(),'runtime_head':P['runtime_head'],'parent':P['prior'],'files':{n:sha(b) for n,b in files.items()},'excluded':['weights','raw videos','full logs','environments','credentials','reference copies'],'runtime_snapshot':before}
name='docs/world-model/task_reward_plan_20261006/two_gpu_b32_manifest.json';files[name]=(json.dumps(manifest,indent=2)+'\n').encode()
for name,raw in files.items():
 path=W/name;assert path.resolve().is_relative_to(W.resolve()) and not path.is_symlink()
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
git(W,'add','--',*sorted(files))
names=git(W,'diff','--cached','--name-only').splitlines();assert set(names)<=set(files) and names
git(W,'diff','--cached','--check')
for name in names:assert sha(subprocess.check_output(['git','-C',str(W),'show',':'+name]))==sha(files[name])
stat=git(W,'diff','--cached','--stat');(E/'reviewed-stat.txt').write_text(stat+'\n')
git(W,'commit','-m','Try one policy GPU and one B32 world-model GPU with preserved sampling')
head=git(W,'rev-parse','HEAD')
assert git(W,'rev-parse','HEAD^')==P['prior'] and snapshot()==before
git(W,'push','personal','HEAD:refs/heads/'+P['branch'])
remote=git(W,'ls-remote','--heads','personal','refs/heads/'+P['branch']).split()
assert remote==[head,'refs/heads/'+P['branch']] and snapshot()==before
receipt={'time':datetime.datetime.now().astimezone().isoformat(),'branch':P['branch'],'commit':head,'verified_remote_commit':remote[0],'pushed':True,'files':names,'runtime_head':P['runtime_head'],'runtime_unchanged':True}
(E/'published.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
if __name__=='__main__':main('--resume-staging' in sys.argv)
