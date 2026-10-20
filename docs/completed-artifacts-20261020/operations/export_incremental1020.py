"""Bounded light artifact delta since the prior full 17 Oct archive."""
import json,os,socket,tarfile,time,datetime
from pathlib import Path
host={'admin':'sz1','h100-gpu02':'sz2','h100-gpu01':'sz3'}[socket.gethostname()]
since=datetime.datetime(2026,10,17,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
O=Path('/data/chenyiteng/exports/completed-light-20261020');O.mkdir(parents=True,exist_ok=True)
roots=[Path('/data/chenyiteng/results'),Path('/home/chenyiteng/results')]
files=[];excluded=[]
for root in roots:
 if not root.exists():continue
 for directory,dirs,names in os.walk(root):
  dirs[:]=[n for n in dirs if n not in {'checkpoints','actor','replay','replay_buffer','videos','video','wandb','ray','tmp','__pycache__'}and not n.startswith(('global_step_','tmp-g','tmp_g'))]
  for name in names:
   p=Path(directory)/name
   if p.is_symlink()or name.startswith(('environment','credential')):continue
   if p.suffix.lower()not in {'.json','.jsonl','.csv','.tsv','.log','.txt','.yaml','.yml','.md','.svg'}and not name.startswith('events.out.tfevents.'):continue
   st=p.stat()
   if st.st_mtime<since:continue
   if st.st_size>8*1024**2:
    excluded.append({'path':str(p),'bytes':st.st_size,'reason':'over 8 MiB single file'});continue
   arc=host+str(p)
   files.append((p,arc,st.st_size,st.st_mtime))
groups={};archives=[];batch=[];size=0
def flush():
 global batch,size
 if not batch:return
 dst=O/f'{host}-light-{len(archives)+1:03d}.tar.gz'
 with tarfile.open(dst,'w:gz')as t:
  for p,arc,_,_ in batch:t.add(p,arcname=arc,recursive=False)
 archives.append({'path':str(dst),'bytes':dst.stat().st_size,'files':len(batch)})
 batch=[];size=0
for row in files:
 if size>80*1024**2:flush()
 batch.append(row);size+=row[2]
flush()
m={'time':time.time(),'host':host,'since':since,'scope':[str(p)for p in roots],
 'files':[{'path':str(p),'archive_path':a,'bytes':n,'mtime':m}for p,a,n,m in files],
 'archives':archives,'large_text_exclusions':excluded,
 'excluded_classes':'weights, optimizer, replay, videos, raw query NPZ/observations, environment/credential files; active files are prefix snapshots'}
(O/f'{host}-manifest.json').write_text(json.dumps(m))
print(json.dumps({'host':host,'files':len(files),'bytes':sum(r[2]for r in files),'archives':archives,'manifest':str(O/f'{host}-manifest.json'),'excluded_large':len(excluded)}))
