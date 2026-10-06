"""Generate the small final-status follow-up after the implementation release."""
import ast,base64,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
DOC='docs/world-model/task_reward_plan_20261006/'
def main(followup=False):
 receipt_name='two_gpu_from0_published.json' if followup else 'two_gpu_b32_published.json'
 receipt=json.loads((ROOT/(DOC+receipt_name)).read_text());assert receipt['pushed'] and receipt['commit']==receipt['verified_remote_commit']
 names=[DOC+'two_gpu_b32_execution.md',DOC+'two_gpu_b32_light.json',DOC+'two_gpu_from0_light.json',DOC+'two_gpu_from0_path_failure.json',DOC+'README.md',DOC+'two_gpu_b32_published.json']
 names += ['local_scripts/lift_two_gpu_20261006/'+n for n in ['make_result_publication.py','status_v2_remote.py','native_offload_remote.py','vector_source_remote.py','fresh_owner.py','prepare_fresh.py','test_resource_timeout.py','make_fresh_package.py','launch_fresh_remote.py','status_fresh_remote.py','build_fresh_light.py']]
 if followup:names.append(DOC+'two_gpu_from0_published.json')
 files={}
 for name in names:
  raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n');assert len(raw)<262144
  if name.endswith('.py'):ast.parse(raw)
  files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 payload={'prior':receipt['commit'],'branch':receipt['branch'],'files':files,'runtime_head':receipt['runtime_head'],
          'release_name':'lift-two-gpu-from0-started-20261007-v1' if followup else 'lift-two-gpu-b32-result-20261006-v1'}
 source='P = '+repr(payload)+'\n'+REMOTE;ast.parse(source)
 (HERE/'result_publication_remote.py').write_text(source,encoding='utf-8',newline='\n')
 print(json.dumps({'files':len(files),'parent':receipt['commit'],'generated_only':True}))
REMOTE=r'''
import base64,hashlib,json,os,socket,subprocess,datetime
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');W=S/'publication/wmrl-bell-release-v1';R=S/'rlinf-opendw-two-gpu-v1'
assert P['release_name'] in ['lift-two-gpu-b32-result-20261006-v1','lift-two-gpu-from0-started-20261007-v1']
E=S/'publication'/P['release_name']
def git(p,*a):return subprocess.check_output(['git','-C',str(p),*a],text=True,timeout=180).strip()
def snapshot():return {'head':git(R,'rev-parse','HEAD'),'status':git(R,'status','--porcelain','--untracked-files=all'),'diff':hashlib.sha256(git(R,'diff','HEAD','--').encode()).hexdigest()}
before=snapshot();assert before['head']==P['runtime_head'] and before['status']==''
F=S/'lift-two-gpu-from0-v2'
for name in ['fresh_owner.py','prepare_fresh.py','test_resource_timeout.py']:
 assert hashlib.sha256((F/'code'/name).read_bytes()).hexdigest()==P['files']['local_scripts/lift_two_gpu_20261006/'+name]['sha256']
assert not E.exists() and git(W,'rev-parse','HEAD')==P['prior'] and git(W,'branch','--show-current')==P['branch']
assert git(W,'remote','get-url','personal')=='git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
assert git(W,'status','--porcelain','--untracked-files=all')==''
assert git(W,'ls-remote','--heads','personal','refs/heads/'+P['branch']).split()==[P['prior'],'refs/heads/'+P['branch']]
E.mkdir(mode=0o700)
for name,row in P['files'].items():
 raw=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 path=W/name;assert path.resolve().is_relative_to(W.resolve()) and not path.is_symlink()
 assert path.suffix in ['.md','.json','.py'] and 0<len(raw)<262144
 path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(raw)
git(W,'add','--',*P['files']);names=git(W,'diff','--cached','--name-only').splitlines();assert names and set(names)<=P['files'].keys()
git(W,'diff','--cached','--check')
for name in names:assert hashlib.sha256(subprocess.check_output(['git','-C',str(W),'show',':'+name])).hexdigest()==P['files'][name]['sha256']
message='Record fresh two-GPU lift rollout and RLT queue' if P['release_name'].startswith('lift-two-gpu-from0-started') else 'Start fresh two-GPU lift WMRL and tolerate optional memory telemetry timeout'
git(W,'commit','-m',message)
head=git(W,'rev-parse','HEAD');assert git(W,'rev-parse','HEAD^')==P['prior'] and snapshot()==before
git(W,'push','personal','HEAD:refs/heads/'+P['branch'])
remote=git(W,'ls-remote','--heads','personal','refs/heads/'+P['branch']).split();assert remote==[head,'refs/heads/'+P['branch']] and snapshot()==before
receipt={'time':datetime.datetime.now().astimezone().isoformat(),'branch':P['branch'],'commit':head,'verified_remote_commit':remote[0],'files':names,'pushed':True,'runtime_head':P['runtime_head'],'runtime_unchanged':True}
(E/'published.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
if __name__=='__main__':
 assert sys.argv[1:] in [[],['--started']]
 main(sys.argv[1:]==['--started'])
