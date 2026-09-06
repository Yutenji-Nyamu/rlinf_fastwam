"""Apply final evidence delta, publish only the dedicated branch and verify ref."""
import datetime,gzip,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/experiment-archive-20260906')
maintenance=Path('/data/chenyiteng/results/server-maintenance-20260906')
secret=sys.stdin.readline().strip().encode();assert secret and os.getuid()==1003
branch='codex/sz-experiment-archive-20260906'
def git(*args,timeout=120,check=True):
 r=subprocess.run(['git','-c','pack.threads=2','-c','core.hooksPath=/dev/null','-C',str(wt),*args],capture_output=True,text=True,timeout=timeout)
 if check:assert r.returncode==0,r.stderr[-3000:]
 return r
assert git('branch','--show-current').stdout.strip()==branch
assert git('status','--porcelain').stdout.strip()==''
assert git('remote','get-url','personal').stdout.strip()=='git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
dest=wt/'evidence/windows';changed=[]
packet_name=sys.argv[1] if len(sys.argv)>1 else 'closeout.tar.gz'
assert packet_name in {'closeout.tar.gz','publication-receipt.tar.gz'}
with tarfile.open(maintenance/packet_name,'r:gz') as tar:
 for item in tar:
  path=PurePosixPath(item.name)
  assert item.isfile() and item.size<50*1024**2 and not path.is_absolute() and '..' not in path.parts and '.git' not in path.parts
  assert path.parts[0] in {'local-workspace','late-final'}
  target=dest.joinpath(*path.parts);assert target.resolve().is_relative_to(dest) and not target.is_symlink()
  data=tar.extractfile(item).read();check=gzip.decompress(data) if target.suffix=='.gz' else data
  assert secret not in check
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);changed.append(str(target.relative_to(wt)))
note='\nThe final documentation/script delta is recorded in `evidence/windows/late-final/`; its hashes override earlier copies in the full manifest.\n'
if note.strip() not in (wt/'README.md').read_text():
 with (wt/'README.md').open('a') as f:f.write(note)
changed.append('README.md')
git('add','-f','--',*changed,timeout=300)
git('commit','-m','Record checkpoint pruning, compact context and BC-DVAC audit closeout',timeout=300)
head=git('rev-parse','HEAD').stdout.strip()
print('PUBLISH_START='+json.dumps({'branch':branch,'head':head,'delta_paths':len(changed)}),flush=True)
# No forced update, no retry on an ambiguous push response.
push=git('push','personal','HEAD:refs/heads/'+branch,timeout=900,check=False)
print('PUSH_RESPONSE='+json.dumps({'rc':push.returncode,'out':push.stdout,'err':push.stderr[-3000:]}),flush=True)
remote=git('ls-remote','--heads','personal','refs/heads/'+branch,timeout=90)
remote_sha=remote.stdout.split()[0] if remote.stdout.split() else None
result={'time':datetime.datetime.now().astimezone().isoformat(),'head':head,'remote_sha':remote_sha,'verified':head==remote_sha,'branch':branch,'files':len(git('ls-files').stdout.splitlines()),'status':git('status','--porcelain').stdout.strip(),
 'df':subprocess.check_output(['df','-B1','/data','/home'],text=True)}
(maintenance/'archive-publish-receipt.json').write_text(json.dumps(result,indent=2))
print('ARCHIVE_PUBLISH='+json.dumps(result),flush=True)
assert result['verified'] and result['status']=='', 'Do not claim publication without matching remote ref'
