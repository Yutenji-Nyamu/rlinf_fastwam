import json,os,time,tarfile
from pathlib import Path
assert os.getuid()==1003
D=Path('/home/chenyiteng/deployment-20261029/single-extend400-sz1-g67')
dest=D/'recent-light-20261029.tar.gz'
assert not dest.exists()
roots=[]
for base in [Path('/home/chenyiteng/results/pi05-distill'),Path('/data/chenyiteng/results/pi05-distill')]:
 for p in base.iterdir():
  if p.is_dir() and (any('202610'+str(day) in p.name for day in range(26,30)) or p.name in ['endpoint-multitask-c10-sz1-g45-20261025','onedp-multitask-c10-sz1-g67-20261025']):roots.append(p)
deployments=[Path('/home/chenyiteng/deployment-20261028')/x for x in ['single-extend200-sz1-g67','dmd-extend200-sz1-g67','mfd-extend200-sz1-g67','physical-v2-formal-sz1-g67']]+[D]
manifest={'snapshot_time':time.time(),'roots':[str(p) for p in roots],'files':[],'oversize':[],'excluded':'weights, videos, replay/images/NPZ, environment variables, control/frame/query bulk logs; stopped job artifacts are included'}
allowed={'.json','.jsonl','.csv','.log','.yaml','.yml','.txt'}
with tarfile.open(dest,'w:gz') as t:
 for root in roots+deployments:
  for directory,dirs,names in os.walk(root):
   dirs[:]=[d for d in dirs if d not in ['source','__pycache__','.git'] and not d.startswith(('tmp-','cache'))]
   for name in names:
    p=Path(directory)/name
    if p.suffix not in allowed or name.startswith('environment-') or any(z in name for z in ['_control','_frames','_controller']) or name.startswith(('query_','u_')) or name in ['queries.json','rlt-plan-before.json']:continue
    s=p.stat()
    if s.st_size>16*2**20:
     manifest['oversize'].append({'path':str(p),'bytes':s.st_size});continue
    arc=str(p).lstrip('/')
    t.add(p,arcname=arc,recursive=False)
    manifest['files'].append({'path':str(p),'bytes':s.st_size,'mtime':s.st_mtime,'archive':arc})
 # Snapshot metadata for reproduceability; live metrics are point-in-time.
 info=tarfile.TarInfo('MANIFEST.json');payload=(json.dumps(manifest,indent=2)+'\n').encode();info.size=len(payload)
 import io
 t.addfile(info,io.BytesIO(payload))
(D/'recent-light-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'archive':str(dest),'bytes':dest.stat().st_size,'files':len(manifest['files']),'raw_bytes':sum(x['bytes'] for x in manifest['files']),'roots':manifest['roots'],'oversize':manifest['oversize']}))
