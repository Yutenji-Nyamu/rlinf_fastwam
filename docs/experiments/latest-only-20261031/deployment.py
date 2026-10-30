import os,json,time,re,stat,importlib.util,tempfile
from pathlib import Path
T=Path('/home/chenyiteng/deployment-20261030/storage-audit-1030f')
assert os.getuid()==1003
spec=importlib.util.spec_from_file_location('ret',T/'retention1031a.py');ret=importlib.util.module_from_spec(spec);spec.loader.exec_module(ret)
with tempfile.TemporaryDirectory() as td:
 p=Path(td)
 for i in [1,2,3]:
  for k in ['checkpoint','policy']:(p/f'{k}-{i:06d}.pt').write_bytes(b'test')
 (p/'checkpoint.json').write_text(json.dumps({'step':3,'path':str(p/'checkpoint-000003.pt'),'policy':str(p/'policy-000003.pt')}))
 (p/'retention-policy.json').write_text('{"mode":"latest_complete_only"}')
 assert len(ret.prune_checkpoints(p,2,[1]))==4
 (p/'checkpoint-000004.pt').write_bytes(b'partial')
 assert ret.prune_checkpoints(p,2,[1])==[] and (p/'checkpoint-000004.pt').exists()
def read(p):return json.loads(p.read_text())
def strings(v):
 if isinstance(v,str):yield v
 elif isinstance(v,dict):
  for x in v.values():yield from strings(x)
 elif isinstance(v,list):
  for x in v:yield from strings(x)
protected=set()
for entry in read(T/'protected1031a.json'):
 if entry.get('role')=='retired_by_user':continue
 for s in strings(entry):
  if s.startswith(('/home/chenyiteng/','/data/chenyiteng/')):protected.add(str(Path(s).resolve()))
deps=[Path('/home/chenyiteng/deployment-20261030/v2-fqlfix400-sz1-g45-recover1'),Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67')]
folders=set();codes=[]
for d in deps:
 plan=read(d/'formal-plan.json');o=Path(read(d/'prepared.json')['output'])
 for s in strings(plan.get('initial_receipts',{})):
  if s.startswith('/'):protected.add(str(Path(s).resolve()))
 for m in plan['methods']:
  p=o/m;p.mkdir(parents=True,exist_ok=True);folders.add(p.resolve())
 codes.append(d/'source/rlinf/algorithms/pi05_distillation/checkpoint_retention.py')
for base in ['/home/chenyiteng/results/pi05-distill','/data/chenyiteng/results/pi05-distill']:
 for p in Path(base).glob('*/*/checkpoint.json'):folders.add(p.parent.resolve())
opened=set()
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=1003:continue
  for s in (p/'cmdline').read_bytes().decode(errors='replace').split('\0'):
   if s.startswith('/'):opened.add(str(Path(s).resolve()))
  for fd in (p/'fd').iterdir():
   try:opened.add(str(Path(os.readlink(fd)).resolve()))
   except OSError:pass
 except OSError:pass
marker={'mode':'latest_complete_only','authorized':'User 2026-10-31: each distillation experiment latest full checkpoint only; logs/evaluations retained','time':time.time()}
targets=[];kept=[];skipped=[]
for p in sorted(folders):
 assert any(p.is_relative_to(Path(b)) for b in ['/home/chenyiteng/results/pi05-distill','/data/chenyiteng/results/pi05-distill'])
 receipt=p/'checkpoint.json'
 if not receipt.exists():continue
 r=read(receipt);step=r['step']
 if not all(Path(r[k]).is_file() and Path(r[k]).resolve()==p/f'{prefix}-{step:06d}.pt' for k,prefix in [('path','checkpoint'),('policy','policy')]):
  skipped.append({'dir':str(p),'reason':'no complete current pair'});continue
 kept.append({'dir':str(p),'step':step})
 for f in p.glob('checkpoint-*.pt'):
  match=re.fullmatch(r'checkpoint-(\d{6})\.pt',f.name)
  if not match or int(match[1])>=step:continue
  pair=[f,p/f'policy-{match[1]}.pt']
  if not all(x.is_file() and not x.is_symlink() for x in pair):continue
  if any(str(x) in protected or str(x) in opened for x in pair):skipped.append({'pair':str(f),'reason':'protected or live reference'});continue
  for x in pair:
   s=x.lstat();assert s.st_uid==1003
   if s.st_nlink!=1:raise RuntimeError('hardlink requires separate review '+str(x))
   targets.append({'path':str(x),'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'allocated':s.st_blocks*512})
record={'time':time.time(),'test':'latest-only preserves complete receipt and unfinished files','kept':kept,'skipped':skipped,'targets':targets,'before_free':{d:os.statvfs(d).f_bavail*os.statvfs(d).f_frsize/2**30 for d in ['/home','/data']}}
assert not (T/'latest1031a-receipt.json').exists()
(T/'latest1031a-plan.json').write_text(json.dumps(record,indent=2))
# Install only the explicitly scoped dynamic override into deployed retention subprocesses.
patch='''    # User-authorized per-run rolling retention, refreshed at each evaluation.
    policy_file = output / "retention-policy.json"
    latest_only = policy_file.is_file() and json.loads(policy_file.read_text()).get("mode") == "latest_complete_only"
    if latest_only:
        keep_last, keep_steps = 1, []
'''
for code in codes:
 s=code.read_text();assert 'latest_only' not in s
 assert '    output = Path(output).resolve()\n' in s
 s=s.replace('    output = Path(output).resolve()\n','    output = Path(output).resolve()\n'+patch,1).replace('(1 if smoke else 2)','(1 if smoke or latest_only else 2)')
 compile(s,str(code),'exec');tmp=code.with_suffix('.py.latest-tmp');tmp.write_text(s);tmp.replace(code)
for p in folders:
 tmp=p/'retention-policy.json.tmp';tmp.write_text(json.dumps(marker));tmp.replace(p/'retention-policy.json')
done=[]
for row in targets:
 p=Path(row['path']);s=p.lstat()
 assert (s.st_ino,s.st_size,s.st_mtime_ns)==(row['inode'],row['size'],row['mtime_ns'])
 p.unlink();done.append(row)
record['deleted_files']=len(done);record['released_GiB']=sum(x['allocated'] for x in done)/2**30
record['after_free']={d:os.statvfs(d).f_bavail*os.statvfs(d).f_frsize/2**30 for d in ['/home','/data']}
record['configured_dirs']=len(folders);record['deployed_sources']=[str(x) for x in codes]
(T/'latest1031a-receipt.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k not in ['targets','kept']}))
