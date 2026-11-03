import json,tarfile,shutil,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
b=Path('/home/chenyiteng/results/pi05-distill/fqlfix-shadow-sz1-20261029/teacher-shadow/batch0')
out=Path('/data/chenyiteng/exports/shadow-pair-raw-20261103');out.mkdir(exist_ok=True)
keys=[]
for f in sorted(b.glob('query_*.npz')):
 if f.stem.endswith('_obs'):continue
 with np.load(f,allow_pickle=False)as z:
  keep={k:z[k] for k in z.files if k.startswith('noise4_')or k.startswith('teacher_')or k in ['before','after','active','u','dv']}
  np.savez_compressed(out/f.name,**keep);keys.append({'query':f.name,'keys':list(keep)})
for n in ['config.json','episodes.json','done.json','validation.json','signals.csv']:
 if(b/n).exists():shutil.copyfile(b/n,out/n)
eps=json.loads((b/'episodes.json').read_text())
for e in eps:
 f=b/e['frame_map'];d=out/e['frame_map'];d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,d)
(out/'README.json').write_text(json.dumps({'source':str(b),'purpose':'Recompute same-position student and teacher entropy/distance; combine teacher4 at e9622c461','arrays':keys,'excludes':'camera observations, full videos and controller binary arrays; retained on source server','time':time.time()},indent=2))
a=out.with_suffix('.tar.gz')
with tarfile.open(a,'w:gz')as t:
 for f in out.rglob('*'):
  if f.is_file():t.add(f,arcname=str(f.relative_to(out)))
print(json.dumps({'archive':str(a),'mib':a.stat().st_size/2**20,'query_count':len(keys),'files':len([f for f in out.rglob('*')if f.is_file()])}))
