"""Import reviewed packet into the dedicated archive tree, then commit (no push)."""
import datetime,gzip,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/experiment-archive-20260906')
maintenance=Path('/data/chenyiteng/results/server-maintenance-20260906')
secret=sys.stdin.readline().strip().encode();assert secret and os.getuid()==1003
def git(*args,timeout=300):
 r=subprocess.run(['git','-c','pack.threads=2','-c','core.hooksPath=/dev/null','-C',str(wt),*args],capture_output=True,text=True,timeout=timeout)
 assert r.returncode==0, r.stderr[-2000:]
 return r.stdout.strip()
assert git('branch','--show-current')=='codex/sz-experiment-archive-20260906'
dest=wt/'evidence/windows';assert dest.is_dir() and not any(dest.iterdir())
count=0;total=0
with tarfile.open(maintenance/'local-evidence.tar.gz','r|gz') as tar:
 for item in tar:
  path=PurePosixPath(item.name)
  assert item.isfile() and item.size<50*1024**2
  assert not path.is_absolute() and '..' not in path.parts and '.git' not in path.parts
  target=dest.joinpath(*path.parts);assert target.resolve().is_relative_to(dest) and not target.exists()
  data=tar.extractfile(item).read();assert len(data)==item.size
  check=gzip.decompress(data) if target.suffix=='.gz' else data
  assert secret not in check, 'Known credential residue in packet'
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  count+=1;total+=len(data)
  if count%10000==0:print('IMPORTED_FILES='+str(count),flush=True)
print('IMPORTED_WINDOWS='+json.dumps({'files':count,'bytes':total}),flush=True)

# Inspect the complete new archive tree before placing it in the Git index.
files=[];opaque=[];large=[];residue=[]
for p in (wt/'evidence').rglob('*'):
 if not p.is_file():continue
 if p.suffix.lower() in {'.pack','.idx','.bundle','.pem','.key','.7z','.rar','.zst'}:opaque.append(str(p))
 if p.stat().st_size>=50*1024**2:large.append(str(p))
 b=p.read_bytes()
 if p.suffix=='.gz':b=gzip.decompress(b)
 if secret in b:residue.append(str(p))
 files.append(p)
assert not opaque and not large and not residue, {'opaque':opaque[:8],'large':large[:8],'residue':residue[:8]}
print('PRECOMMIT_REVIEW='+json.dumps({'files':len(files),'large_files':len(large),'credential_residual_files':len(residue)}),flush=True)
git('add','-f','--','README.md','evidence',timeout=600)
print('INDEX_READY',flush=True)
git('commit','-m','Archive owned RL experiment code and lightweight results; exclude replay, videos and weights',timeout=600)
commit=git('rev-parse','HEAD')
stats=git('count-objects','-v')
result={'time':datetime.datetime.now().astimezone().isoformat(),'commit':commit,'branch':git('branch','--show-current'),'indexed_files':len(git('ls-files').splitlines()),'git_objects':stats,'status':git('status','--porcelain')}
(maintenance/'archive-commit-result.json').write_text(json.dumps(result,indent=2))
print('ARCHIVE_COMMIT='+json.dumps(result),flush=True)
