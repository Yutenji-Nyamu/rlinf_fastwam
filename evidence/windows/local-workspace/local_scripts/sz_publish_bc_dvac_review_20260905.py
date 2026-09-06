"""Executed on server with explicit pre-reviewed PACKAGES; evidence-only writes."""
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

assert os.getuid()==1003
base=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
def git(tree,*args):
    return subprocess.check_output(['git','-C',str(tree),*args],text=True,timeout=50).strip()
for package in PACKAGES:
    tree=base/package['tree'];head=package['expected'];branch=package['branch']
    assert git(tree,'rev-parse','HEAD')==head
    assert git(tree,'branch','--show-current')==branch
    assert not git(tree,'status','--porcelain','--untracked-files=all')
    target=tree/'evidence/bc_dvac_review_20260905'
    assert not target.exists()
    manifest=[]
    for rel,value in package['files'].items():
        p=target/rel
        assert p.is_relative_to(target) and '..' not in p.parts
        raw=base64.b64decode(value)
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        manifest.append({'path':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    (target/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    git(tree,'add','-f','--',str(target.relative_to(tree)))
    staged=git(tree,'diff','--cached','--name-only').splitlines()
    assert staged and all(p.startswith('evidence/bc_dvac_review_20260905/') for p in staged)
    git(tree,'commit','-m','Archive online BC DVAC discussion and current experiment evidence')
    git(tree,'push','personal',f'HEAD:refs/heads/{branch}')
    new=git(tree,'rev-parse','HEAD')
    assert git(tree,'ls-remote','personal',f'refs/heads/{branch}').split()[0]==new
    assert not git(tree,'status','--porcelain')
    changed=git(tree,'diff','--name-only',head,new).splitlines()
    assert all(p.startswith('evidence/bc_dvac_review_20260905/') for p in changed)
    print(json.dumps({'branch':branch,'head':new,'source_unchanged':True,'files':len(changed),'bytes':sum(p['bytes'] for p in manifest),'remote_matches':True}),flush=True)
tree=base/'pi0-online-bc'
remote=dict((ref,head) for head,ref in (x.split() for x in git(tree,'ls-remote','personal','refs/heads/codex/*').splitlines()))
checked=[]
for line in git(tree,'for-each-ref','--format=%(refname) %(objectname)','refs/heads/codex/').splitlines():
    ref,head=line.split();checked.append({'branch':ref,'local':head,'remote':remote.get(ref),'matches':remote.get(ref)==head})
print('FINAL_RLINF_BRANCH_REFS '+json.dumps(checked),flush=True)
assert all(p['matches'] for p in checked)
