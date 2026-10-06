"""Publish only C20 operations and light receipts in an independent source checkout."""
import hashlib,json,subprocess,time,zipfile
from pathlib import Path
control=Path('/data/chenyiteng/deployment-20261006/dsrl-u-c20-lease-v2')
p=json.loads((control/'plan.json').read_text());active=Path(p['dsrl_repo'])
repo=active.with_name('dsrl-c20-evidence-20261006')
branch='codex/dsrl-c20-restart-20261006'
expected='8bcd99a6df38a1700a988d8982bde9368bf18025'
def git(r,*args):
    return subprocess.check_output(['git','-C',str(r),*args],text=True,stderr=subprocess.PIPE,timeout=120).strip()
assert git(active,'rev-parse','HEAD')==expected and not git(active,'status','--porcelain')
acceptance=json.loads((control/'c20-acceptance.json').read_text())
assert acceptance['protected_unchanged']
assert all(r['state']=='FORMAL' for r in acceptance['state']['slots'].values())
assert not repo.exists()
git(active,'worktree','add','-b',branch,str(repo),expected)
names=[]
with zipfile.ZipFile(control/'c20-delivery.zip') as archive:
    for item in archive.infolist():
        rel=Path(item.filename)
        assert not rel.is_absolute() and '..' not in rel.parts
        assert rel.parts[0] in ('docs','tools') and item.file_size<1000000
        f=repo/rel;assert not f.exists()
        f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(archive.read(item));names.append(item.filename)
dest=repo/'docs/dsrl_c20_20261006';dest.mkdir(parents=True,exist_ok=True)
for name in ('c20-configs.json','c20-cpu-check.json','c10-released.json','c20-formal-started.json','c20-acceptance.json'):
    f=dest/name;f.write_bytes((control/name).read_bytes());names.append(f.relative_to(repo).as_posix())
for role in ('clean','u'):
    run=Path('/data/chenyiteng/results/rlinf-dsrl-pi05-u-20261006')/(role+'-c20-formal-200-mb256-v2')
    f=dest/(role+'-resolved.yaml');f.write_bytes((run/'runtime/resolved.yaml').read_bytes());names.append(f.relative_to(repo).as_posix())
assert len(names)==len(set(names))
dirty=set(git(repo,'ls-files','--others','--exclude-standard').splitlines())
assert dirty==set(names)
git(repo,'add','--',*names)
assert set(git(repo,'diff','--cached','--name-only').splitlines())==set(names)
git(repo,'diff','--cached','--check')
git(repo,'commit','-m','Restart DSRL Clean and U with C20 from original SFT')
head=git(repo,'rev-parse','HEAD')
git(repo,'push','personal',head+':refs/heads/'+branch)
assert git(repo,'ls-remote','personal','refs/heads/'+branch).split()[0]==head
assert git(active,'rev-parse','HEAD')==expected and not git(active,'status','--porcelain')
out={'time':time.time(),'head':head,'branch':branch,'source_head':expected,
     'files':{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names}}
with (control/'c20-published.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='files'}))
