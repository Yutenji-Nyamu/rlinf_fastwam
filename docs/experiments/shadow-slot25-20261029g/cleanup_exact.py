import json,os,stat,time
from pathlib import Path
assert os.getuid()==1003 and os.uname().nodename=='admin'
D=Path('/home/chenyiteng/deployment-20261029/fqlfix-shadow-sz1')
receipt=D/'cleanup-1029g-receipt.json'
assert not receipt.exists(),'Operation already recorded; inspect receipt instead of repeating'
read=lambda p:json.loads(Path(p).read_text())
manifest=read(D/'cleanup-proposal-1029g.json')
approved={'已完成FQL-fix smoke','已停旧CP四个保留点','已停旧FQL最佳和末代'}
groups=[g for g in manifest['candidates'] if g['name'] in approved]
assert len(groups)==3
targets=[r for g in groups for r in g['files']]
assert len(targets)==16 and len({r['path'] for r in targets})==16
N=Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67')
plan=read(N/'formal-plan.json')
assert plan['fresh_fql_fix'] and 'cp' not in plan['methods'] and 'cp_v2' in plan['methods']
refs=[read(D/'protected-registry-1029g.json'),plan]
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
rlt=read(S/'plan.json');refs.append(rlt)
for slot in rlt['slots'].values():refs.append(read(slot['fallback']))
references=json.dumps(refs)
names={r['path'] for r in targets};opened={}
for proc in Path('/proc').glob('[0-9]*'):
    try:
        if proc.stat().st_uid!=1003:continue
        for fd in (proc/'fd').iterdir():
            try:
                link=os.readlink(fd)
                if link in names:opened[link]=int(proc.name)
            except (FileNotFoundError,PermissionError):pass
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not opened,opened
for r in targets:
    p=Path(r['path']);s=p.lstat()
    assert p.is_relative_to('/home/chenyiteng/results/pi05-distill') and p.resolve()==p
    assert '/protected-eval/' not in str(p) and str(p) not in references
    assert stat.S_ISREG(s.st_mode) and s.st_uid==1003 and s.st_nlink==1
    assert s.st_size==r['size'] and s.st_ino==r['inode'] and s.st_dev==r['dev']
def disk(p):
    s=os.statvfs(p);return {'available':s.f_bavail*s.f_frsize,'used':(s.f_blocks-s.f_bfree)*s.f_frsize}
result={'operation':'user-approved 13GiB smoke plus 45GiB retired CP/FQL large files only','time_started':time.time(),
        'before':{p:disk(p) for p in ['/home','/data']},'deleted':[],'groups':[{'name':g['name'],'allocated_bytes':sum(r['allocated'] for r in g['files'])} for g in groups],
        'kept':'directories, logs, curves, small Q diagnostics, replay, current/queued complete resume points, protected weights and RLT dependencies'}
with (D/'cleanup-1029g-started.json').open('x') as f:json.dump(result,f)
for r in targets:
    Path(r['path']).unlink();result['deleted'].append(r)
result.update(time_finished=time.time(),after={p:disk(p) for p in ['/home','/data']},allocated_bytes=sum(r['allocated'] for r in targets))
assert not any(Path(r['path']).exists() for r in targets)
receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='deleted'},ensure_ascii=False))
