"""Runs on Linux only. PACKET comes from the frozen, explicit approved file list."""
import datetime
import hashlib
import os
import pickle
import re
import stat
import subprocess
import zipfile
from pathlib import Path

BASE=Path('/data/chenyiteng/results/rlinf-shenzhen')
UID=os.getuid()
assert UID==1003 and BASE.resolve()==BASE
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def metadata(p):
    p=Path(p);s=p.lstat()
    assert p.is_absolute() and p.is_relative_to(BASE) and p.resolve()==p and not p.is_symlink(),str(p)
    assert stat.S_ISREG(s.st_mode) and s.st_uid==UID,str(p)
    return {'path':str(p),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'inode':s.st_ino,'device':s.st_dev,'mtime_ns':s.st_mtime_ns,'nlink':s.st_nlink,'uid':s.st_uid}
def same(item):
    current=metadata(item['path'])
    assert all(current[k]==item[k] for k in ('bytes','inode','device','mtime_ns','nlink','uid')),item['path']
    return current
def space():
    s=os.statvfs(BASE);return {'available_bytes':s.f_bavail*s.f_frsize,'free_bytes':s.f_bfree*s.f_frsize}
def checkpoint(p):
    p=Path(p);files=[metadata(q) for q in sorted(p.rglob('*')) if q.is_file()]
    assert files,p
    local=list(p.glob('actor/local_shard_checkpoint/checkpoint_rank_*.pt'))
    full=p/'actor/model_state_dict/full_weights.pt'
    dcp=p/'actor/dcp_checkpoint/.metadata'
    status=None
    if dcp.is_file():
        with dcp.open('rb') as f:meta=pickle.load(f)
        refs={str(x.relative_path) for x in meta.storage_data.values()}
        assert refs and all((dcp.parent/x).is_file() for x in refs),('DCP references missing',str(p),refs)
        for x in meta.storage_data.values():assert (dcp.parent/x.relative_path).stat().st_size>=x.offset+x.length
        status='DCP metadata references and byte extents complete; no model restore'
    elif local:
        expected=2 if '2gpu' in str(p) else (4 if '4gpu' in str(p) else 1)
        assert {int(re.search(r'rank_(\d+)',q.name)[1]) for q in local}==set(range(expected)),('Missing native rank',str(p))
        status='Native expected rank files present; no model restore'
    elif full.is_file() and 'smoke' in str(p):
        status='LEGACY_SMOKE_MODEL_ONLY: training shards already removed Sep1; preserve remaining full model'
    else:raise AssertionError(('No complete retained training structure',str(p)))
    for q in [*local,*([full] if full.is_file() else [])]:
        with zipfile.ZipFile(q) as z:
            names=z.namelist();assert any(n.endswith('/data.pkl') for n in names) and any('/data/' in n for n in names),str(q)
    return {'path':str(p),'validation':status,'files':files}
def active_checks(candidates):
    target_paths={x['path'] for x in candidates}
    target_gens={str(Path(x['path']).parents[2]) if '/dcp_checkpoint/' in x['path'] or '/model_state_dict/' in x['path'] or '/local_shard_checkpoint/' in x['path'] else x['path'] for x in candidates}
    opened=[];active=[];config_refs=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=UID:continue
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'train_embodied_agent.py' in cmd:
                assert not any(t in cmd for t in target_paths|target_gens),('Active resume target',proc.name)
                active.append({'pid':int(proc.name),'cmd':cmd})
                hit=re.search(r'runner.logger.log_path=(\S+)',cmd)
                if hit:
                    run=Path(hit[1])
                    for config in run.rglob('*'):
                        if not config.is_file() or config.suffix not in ('.yaml','.json','.sh') or config.stat().st_size>2_000_000:continue
                        if 'checkpoints' in config.parts or 'video' in config.parts:continue
                        text=config.read_text(errors='replace')
                        matches=[t for t in target_gens if t in text]
                        assert not matches,('Active run config references target',str(config),matches)
            for fd in (proc/'fd').iterdir():
                try:
                    link=os.readlink(fd)
                    if link in target_paths:opened.append({'pid':int(proc.name),'path':link})
                except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert not opened,('Target is open',opened)
    # Historical resume references are evidence, not active blockers. Examine small run configs only.
    for category in ('grpo','ppo','pi05','fastwam-grpo','online-bc'):
        root=BASE/category
        for folder,dirs,names in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('checkpoints','video','.git','ray','online_bc','success_data') and not (Path(folder)/d).is_symlink()]
            for name in names:
                p=Path(folder)/name
                if p.suffix not in ('.yaml','.json','.sh') or p.is_symlink() or p.stat().st_size>2_000_000:continue
                text=p.read_text(errors='replace')
                if 'resume_dir' not in text and 'ckpt_path' not in text:continue
                matches=[t for t in target_gens if t in text]
                if matches:config_refs.append({'config':str(p),'targets':matches})
    return active,config_refs

if PACKET['mode']=='verify':
    audit=Path(PACKET['audit_path'])
    result=json.loads((audit/'result.json').read_text())
    missing=[x['path'] for x in result['deleted'] if not Path(x['path']).exists()]
    assert len(missing)==PACKET['count']
    for f in PACKET['protected_files']+PACKET['small_files']:same(f)
    print(json.dumps({'verified_at':now(),'deleted_absent':len(missing),'protected_unchanged':len(PACKET['protected_files']),'small_files_unchanged':len(PACKET['small_files']),'space':space(),'audit_path':str(audit)}))
else:
    started=now();before=space();candidates=[]
    for old in PACKET['candidate_files']:
        current=metadata(old['path'])
        assert current['bytes']==old['bytes'] and current['bytes']>=1024**3 and current['nlink']==1,current
        assert not any(s in current['path'] for s in ('/pi05-sidney/','/rlinf-current-dsrl/','/rlinf-rlt/','-v8/')),current
        if PACKET['mode']=='execute':same(old)
        candidates.append({**old,**current})
    assert len({x['path'] for x in candidates})==len(candidates)
    keeps=[];invalid_keeps=[]
    for k in PACKET['keeps']:
        try:keeps.append({**k,**checkpoint(k['path'])})
        except (AssertionError,FileNotFoundError,zipfile.BadZipFile) as e:
            invalid_keeps.append({'root':k['root'],'path':k['path'],'reason':str(e)})
    if PACKET['mode']=='execute':assert not invalid_keeps,invalid_keeps
    bad_roots={k['root'] for k in invalid_keeps}
    deferred=[x for x in candidates if x.get('run_root') in bad_roots]
    candidates=[x for x in candidates if x.get('run_root') not in bad_roots]
    protected=[f for k in keeps for f in k['files']]
    assert not ({x['path'] for x in candidates}&{x['path'] for x in protected})
    small=[];seen=set()
    for x in candidates:
        folder=Path(x['path']).parents[2]
        if str(folder) in seen:continue
        seen.add(str(folder))
        for q in folder.rglob('*'):
            if q.is_file() and q.stat().st_size<1024**3:small.append(metadata(q))
    active,refs=active_checks(candidates)
    value={'passed':True,'time_started':started,'time_checked':now(),'candidate_files':candidates,'keeps':keeps,'protected_files':protected,'small_files':small,'active_processes':active,'references':refs,'count':len(candidates),'allocated_gib':sum(x['allocated_bytes'] for x in candidates)/1024**3,'space_before':before,'audit_path':PACKET['audit_path'],'invalid_keeps':invalid_keeps,'deferred':deferred,'deferred_count':len(deferred),'deferred_gib':sum(x['allocated_bytes'] for x in deferred)/1024**3,'scope':'Validated subset of 107 explicitly approved files >=1GiB; no directory removal; no current Sidney/DSRL/RLT/v8/other users'}
    if PACKET['mode']=='execute':
        for f in PACKET['protected_files']+PACKET['small_files']:same(f)
        audit=Path(PACKET['audit_path']);assert not audit.exists(),'Audit already exists; investigate, never replay'
        audit.mkdir()
        (audit/'preflight.json').write_text(json.dumps(value,indent=2))
        deleted=[]
        with (audit/'deletions.jsonl').open('x') as ledger:
            for item in candidates:
                same(item)
                # The only destructive operation: an explicit regular file, never a directory/glob.
                os.unlink(item['path'])
                row={**item,'deleted_at':now()};deleted.append(row)
                ledger.write(json.dumps(row)+'\n');ledger.flush();os.fsync(ledger.fileno())
        for f in protected+small:same(f)
        assert all(not Path(x['path']).exists() for x in deleted)
        value.update({'deleted':deleted,'deleted_count':len(deleted),'finished_at':now(),'space_after':space(),'protected_unchanged':len(protected),'small_files_unchanged':len(small)})
        (audit/'result.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value))
