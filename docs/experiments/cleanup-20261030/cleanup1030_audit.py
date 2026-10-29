import json,os,re,stat,time,sys,shutil
from pathlib import Path
assert os.getuid()==1003 and os.uname().nodename=='admin'
T=Path('/home/chenyiteng/deployment-20261030/queue-storage-sz1')
read=lambda p:json.loads(Path(p).read_text())
old=read(T/'cleanup-candidates.json');registry=read(T/'protected-registry.json')
groups=[g for g in old['candidates'] if g['title'] in ['Home：旧单任务100轮点','Data：Direct C10早期评估点']]
assert len(groups)==2
explicit=[x for g in groups for x in g['files']];assert len(explicit)==16
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');rlt=read(S/'plan.json')
refs=[registry,rlt]+[read(s['fallback']) for s in rlt['slots'].values()]
for D in [Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67'),Path('/home/chenyiteng/deployment-20261028/physical-v2-formal-sz1-g67')]:
 refs.extend(read(D/f) for f in ['formal-plan.json','training-queue-status.json'])
opened=set();commands=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=1003:continue
  commands.append((p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
  for fd in (p/'fd').iterdir():
   try:opened.add(os.readlink(fd))
   except OSError:pass
 except (OSError,ProcessLookupError):pass
references=json.dumps(refs)+'\n'.join(commands)
def info(p):
 s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_uid==1003
 return dict(path=str(p),size=s.st_size,allocated=s.st_blocks*512,nlink=s.st_nlink,inode=s.st_ino,dev=s.st_dev,mtime=s.st_mtime)
for x in explicit:
 now=info(Path(x['path']))
 assert all(now[k]==x[k] for k in ['size','inode','dev','nlink'])
 assert now['nlink']==1 and x['path'] not in references and x['path'] not in opened
extra=[];inventory=[]
for base in ['/home/chenyiteng/results/pi05-distill','/data/chenyiteng/results/pi05-distill']:
 for run in Path(base).iterdir():
  if not run.is_dir() or 'smoke' in run.name:continue
  for d in run.iterdir():
   if not d.is_dir():continue
   cps=sorted(d.glob('checkpoint-*.pt'))
   if not cps:continue
   steps=[int(p.stem.split('-')[-1]) for p in cps]
   b=run/f'best-{d.name}.json';best=read(b).get('checkpoint',{}).get('step') if b.exists() else None
   keep=set(steps[-2:]);complete=False
   if run.name=='physical-v2-formal-sz1-g67-20261028':
    c=read(Path('/home/chenyiteng/deployment-20261028/physical-v2-formal-sz1-g67')/(d.name+'.json'))
    final=max(c['evaluation_steps']);sm=run/f'summary-{d.name}-s{final:06d}.json'
    complete=sm.exists() and read(sm).get('complete',False)
    if complete:keep={final}
   if best is not None:keep.add(best)
   inventory.append(dict(run=str(run),method=d.name,steps=steps,best=best,keep=sorted(keep),completed=complete))
   if best is None:continue
   for p in cps:
    step=int(p.stem.split('-')[-1])
    if step in keep:continue
    pair=[p,d/f'policy-{step:06d}.pt']
    if not all(z.exists() for z in pair):continue
    rr=[info(z) for z in pair]
    if any(x['path'] in references or x['path'] in opened or x['nlink']!=1 or x['size']<64*1024**2 or '/protected-eval/' in x['path'] for x in rr):continue
    if any(x['path'] in {v['path'] for v in explicit} for x in rr):continue
    extra.extend(rr)
out=dict(time=time.time(),explicit=explicit,extra=extra,inventory=inventory,free_gib={p:shutil.disk_usage(p).free/2**30 for p in ['/home','/data']})
(T/'cleanup1030-plan.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
