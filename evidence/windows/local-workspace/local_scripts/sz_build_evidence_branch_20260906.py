"""Create a dedicated evidence branch. Do not commit or push until separate review."""
import datetime, hashlib, json, os, subprocess, sys, tarfile
from pathlib import Path, PurePosixPath
from lightweight_archive_20260906 import Archive

root = Path('/data/chenyiteng/projects/rlinf-shenzhen/RLinf')
wt = root.parent/'worktrees/experiment-archive-20260906'
branch = 'codex/sz-experiment-archive-20260906'
maintenance = Path('/data/chenyiteng/results/server-maintenance-20260906')
packet = maintenance/'local-evidence.tar.gz'
secret = sys.stdin.readline().strip()
assert secret and os.getuid() == 1003

def git(path, *args, timeout=90, check=True):
    r = subprocess.run(['git','-C',str(path),*args],capture_output=True,timeout=timeout)
    if check and r.returncode: raise RuntimeError('Git failed '+repr(args)+': '+r.stderr.decode(errors='replace')[-1000:])
    return r

assert git(root,'remote','get-url','personal').stdout.decode().strip() == 'git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
assert not wt.exists()
assert git(root,'show-ref','--verify','refs/heads/'+branch,check=False).returncode != 0
git(root,'worktree','add','--detach',str(wt),'HEAD')
git(wt,'switch','--orphan',branch)
print('CREATED_EMPTY_EVIDENCE_BRANCH='+branch,flush=True)

dest = wt/'evidence'
arc = Archive(dest,secret)
arc.walk('/data/chenyiteng/results','server-results',skip=(str(maintenance),))
print('SERVER_RESULTS_INCLUDED='+str(arc.counts['included']),flush=True)
git_inventory=[]
for repo in (root, root.parent/'RoboTwin-RLinf-support'):
    listing=git(repo,'worktree','list','--porcelain').stdout.decode()
    for block in listing.strip().split('\n\n'):
        values=dict(l.split(' ',1) for l in block.splitlines() if ' ' in l)
        if 'worktree' not in values:continue
        tree=Path(values['worktree'])
        if tree==wt or not tree.exists():continue
        status=git(tree,'status','--porcelain=v1').stdout.decode(errors='replace')
        refs=git(tree,'remote','-v').stdout.decode(errors='replace')
        diff=git(tree,'diff','HEAD','--binary').stdout
        prefix='source-snapshots/'+tree.name
        # All tracked source/records plus untracked files, filtered through the same policy.
        tracked=git(tree,'ls-files','-z').stdout.decode().split('\0')
        untracked=git(tree,'ls-files','--others','--exclude-standard','-z').stdout.decode().split('\0')
        for name in sorted(set(tracked+untracked)-{''}):
            p=tree/name
            if p.is_file() and not p.is_symlink():arc.file(p,prefix+'/'+name)
        if diff:arc.add(str(tree)+' git diff HEAD', 'uncommitted/'+tree.name+'.patch',diff)
        git_inventory.append({'path':str(tree),'head':values.get('HEAD'),'branch':values.get('branch'),'status':status,'remotes':refs,'tracked':len(tracked)-1,'untracked':len(untracked)-1})
arc.add('current owned worktrees','git-worktrees.json',json.dumps(git_inventory,indent=2).encode())
arc.finish({'scope':'Own Shenzhen results and all tracked/untracked source of known worktrees; original dirty files untouched; active logs are point-in-time copies'})

# Packet was filtered locally; validate path/type/size and extract regular files only.
windows=dest/'windows'
windows.mkdir()
members=0
if packet.exists():
 with tarfile.open(packet,'r:gz') as tar:
    for member in tar:
        p=PurePosixPath(member.name)
        assert member.isfile() and not p.is_absolute() and '..' not in p.parts and '.git' not in p.parts
        assert member.size < 50*1024**2
        q=windows.joinpath(*p.parts);q.parent.mkdir(parents=True,exist_ok=True)
        assert not q.exists()
        with q.open('xb') as out:
            src=tar.extractfile(member)
            while True:
                data=src.read(1024**2)
                if not data:break
                out.write(data)
        members+=1
print('WINDOWS_PACKET_FILES='+str(members),flush=True)
(wt/'README.md').write_text('''# RL experiment evidence archive — 2026-09-06

This artifact-only branch archives the owned Shenzhen results and Windows rl workspace.
It is NOT a training/runtime branch and does not change any running experiment.

- `evidence/server-results/`: point-in-time configurations, logs, metrics, plots, tiny sidecars.
- `evidence/source-snapshots/`: source/records from known server worktrees, including local edits.
- `evidence/uncommitted/`: unvalidated diagnostic patches, preserved as evidence only.
- `evidence/windows/`: local historical notes, scripts, results, filtered container members.
- Each archive manifest records included/excluded files, hashes, redactions and compression.

Excluded by explicit user choice: replay/raw tensor datasets even when each shard is small;
videos, model/checkpoint weights, large binaries, caches and credentials. Large text is
losslessly gzip-compressed when practical. Original local/remote files are not rewritten.
This is an evidence backup, not a complete model/data backup or proof of resumability.
Recent still-running logs reflect the copy time, not final experiment completion.
''',encoding='utf-8')
summary={'time':datetime.datetime.now().astimezone().isoformat(),'branch':branch,'worktree':str(wt),'server_files':arc.counts['included'],'windows_packet_files':members,'server_bytes':arc.bytes['included'],'dirty_snapshot_trees':[g['path'] for g in git_inventory if g['status']]}
(maintenance/'archive-build-result.json').write_text(json.dumps(summary,indent=2))
print('ARCHIVE_BUILD='+json.dumps(summary),flush=True)
