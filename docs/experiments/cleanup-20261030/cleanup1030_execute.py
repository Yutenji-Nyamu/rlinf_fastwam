import json,os,stat,time,shutil
from pathlib import Path
assert os.getuid()==1003 and os.uname().nodename=='admin'
T=Path('/home/chenyiteng/deployment-20261030/queue-storage-sz1');read=lambda p:json.loads(Path(p).read_text())
receipt=T/'cleanup1030-receipt.json';assert not receipt.exists()
plan=read(T/'cleanup1030-plan.json');rows=plan['explicit']+plan['extra']
assert len(rows)==18 and all('/cp_v2/' in x['path'] and '-005400.pt' in x['path'] for x in plan['extra'])
D=Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67')
q=read(D/'formal-plan.json');refs=[q,read(D/'training-queue-status.json'),read(T/'protected-registry.json')]
V=Path('/home/chenyiteng/deployment-20261028/physical-v2-formal-sz1-g67');refs.append(read(V/'training-queue-status.json'))
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');rp=read(S/'plan.json');refs.append(rp)
for sl in rp['slots'].values():refs.append(read(sl['fallback']))
for r in q['initial_receipts'].values():
 assert all(Path(r[k]).is_file() and Path(r[k]).stat().st_size>64*1024**2 for k in ['path','policy'])
C=Path('/data/chenyiteng/results/pi05-distill/physical-v2-formal-sz1-g67-20261028')
assert read(C/'best-cp_v2.json')['checkpoint']['step']==3000
assert read(C/'summary-cp_v2-s006000.json')['complete']
refs_text=json.dumps(refs);names={x['path'] for x in rows};assert len(names)==18
for proc in Path('/proc').glob('[0-9]*'):
 try:
  if proc.stat().st_uid!=1003:continue
  cmd=(proc/'cmdline').read_bytes().decode(errors='replace')
  assert not any(p in cmd for p in names)
  for fd in (proc/'fd').iterdir():
   try:assert os.readlink(fd) not in names
   except OSError:pass
 except (OSError,ProcessLookupError):pass
for r in rows:
 p=Path(r['path']);st=p.lstat()
 assert any(p.is_relative_to(root) for root in ['/home/chenyiteng/results/pi05-distill','/data/chenyiteng/results/pi05-distill']) and p.resolve()==p
 assert str(p) not in refs_text and '/protected-eval/' not in str(p)
 assert stat.S_ISREG(st.st_mode) and st.st_uid==1003 and st.st_nlink==1
 assert (st.st_ino,st.st_dev,st.st_size)==(r['inode'],r['dev'],r['size'])
free=lambda:{p:shutil.disk_usage(p).free/2**30 for p in ['/home','/data']}
result=dict(time_started=time.time(),before=free(),deleted=[],kept='all small files/logs/curves/directories; 200-epoch resume points; current latest2; protected models and RLT dependencies')
with (T/'cleanup1030-journal.jsonl').open('x') as log:
 for r in rows:
  Path(r['path']).unlink();result['deleted'].append(r);log.write(json.dumps(r)+'\n');log.flush()
result.update(time_finished=time.time(),after=free(),allocated_gib=sum(r['allocated'] for r in rows)/2**30)
receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='deleted'}))
