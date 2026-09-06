"""Read-only inventory of the requesting user's Shenzhen Git/run evidence."""
import datetime
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

assert os.getuid() == 1003
base = Path('/data/chenyiteng/projects/rlinf-shenzhen')

def git(root, *args, timeout=20):
    p = subprocess.run(['git','--no-optional-locks','-C',str(root),*args], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def safe_url(url):
    if '://' not in url: return url
    u=urlsplit(url)
    return urlunsplit((u.scheme,(u.hostname or '')+(':'+str(u.port) if u.port else ''),u.path,'',''))

paths=[p for p in base.iterdir() if p.is_dir() and (p/'.git').exists()]
paths += [p for p in (base/'worktrees').iterdir() if p.is_dir() and (p/'.git').exists()]
for seed in list(paths):
    _,listing,_=git(seed,'worktree','list','--porcelain')
    for line in listing.splitlines():
        if line.startswith('worktree '):
            linked=Path(line[9:])
            if str(linked).startswith('/data/chenyiteng/') and linked.exists(): paths.append(linked)
repos={}
trees=[]
covered={}
for root in sorted(set(paths)):
    rc,common,_=git(root,'rev-parse','--path-format=absolute','--git-common-dir')
    if rc: continue
    if common not in repos:
        _,remotes,_=git(root,'remote')
        remote_meta={}
        for remote in remotes.splitlines():
            _,url,_=git(root,'remote','get-url',remote)
            entry={'url':safe_url(url)}
            if remote == 'personal':
                try:
                    code,out,err=git(root,'-c','http.lowSpeedLimit=1024','-c','http.lowSpeedTime=12','ls-remote','--heads',remote,timeout=25)
                    entry.update(rc=code,heads={line.split()[1][11:]:line.split()[0] for line in out.splitlines() if len(line.split())==2})
                    if code: entry['error']='live ls-remote failed; no remote verification'
                except subprocess.TimeoutExpired:
                    entry.update(rc=124,error='live ls-remote timeout; no remote verification')
            remote_meta[remote]=entry
        _,branchtext,_=git(root,'for-each-ref','--format=%(refname:short)|%(objectname)|%(upstream:short)','refs/heads')
        branches=[]
        for line in branchtext.splitlines():
            name,head,upstream=line.split('|')
            remotehead=remote_meta.get('personal',{}).get('heads',{}).get(name)
            branches.append(dict(name=name,head=head,upstream=upstream,personal_head=remotehead,exact_remote_match=(head==remotehead)))
        repos[common]=dict(root=str(root),remotes=remote_meta,branches=branches)
    _,head,_=git(root,'rev-parse','HEAD')
    _,branch,_=git(root,'branch','--show-current')
    _,dirty,_=git(root,'status','--porcelain','--untracked-files=all')
    _,tracked,_=git(root,'ls-files','evidence')
    evidence=[]
    for rel in tracked.splitlines():
        file=root/rel
        if file.is_file():
            evidence.append((rel,file.stat().st_size))
            if file.name == 'SERVER_RUN_PATH.txt':
                run=file.read_text().strip()
                covered.setdefault(run,[]).append({'branch':branch,'file':str(file)})
    trees.append(dict(root=str(root),common=common,branch=branch,head=head,dirty=dirty.splitlines(),evidence_files=len(evidence),evidence_bytes=sum(x[1] for x in evidence)))

runs=[]
result=Path('/data/chenyiteng/results')
prune={'checkpoints','video','videos','robotwin_data','ray_logs','ray','success_data','exports','bundles','packets','assets'}
seen=set()
for start in [result/'rlinf-shenzhen', result/'rlinf-rlt', result/'rlinf-dsrl', result/'rlinf-rlt-dvac-pure']:
    if not start.exists(): continue
    for directory,dirs,files in os.walk(start,followlinks=False):
        path=Path(directory)
        depth=len(path.relative_to(start).parts)
        dirs[:]=[d for d in dirs if d not in prune and (depth < 5)]
        isrun=('driver.log' in files or 'exit_code.txt' in files or 'exit_code' in files)
        if path.parent.name=='runs': isrun=True
        if not isrun or any(parent in seen for parent in path.parents): continue
        seen.add(path)
        exits={str(p.relative_to(path)):p.read_text(errors='replace')[:160] for folder in [path,path/'runtime'] for n in ('exit_code.txt','exit_code','finished_at.txt') if (p:=folder/n).is_file()}
        cov=covered.get(str(path),[])
        if not cov:
            for parent in path.parents:
                if str(parent) in covered: cov=covered[str(parent)]; break
        runs.append(dict(path=str(path),exit_metadata=exits,covered_by=cov))

print('GIT_COVERAGE_JSON '+json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),repositories=repos,worktrees=trees,runs=runs),ensure_ascii=False))
