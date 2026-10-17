"""Read metadata of owned result artifacts; no tensors loaded, no GPU context."""
import os,json,time,datetime,collections
from pathlib import Path
os.nice(15);uid=os.getuid();cutoff=datetime.datetime(2026,9,26,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
out={'time':time.time(),'cutoff':'2026-09-26T00:00:00+08:00','files':[],'weights':[],'links':[],'errors':[],'roots':[]}
seen=set();skip={'.git','__pycache__','replay','replay_buffer','replay_buffers','success_replay','online','dataset','datasets','videos','video','images','frames','wandb','media','data','assets','models','source','src','venv','.venv','site-packages','node_modules','third_party'}
roots=[p for p in Path('/data/chenyiteng/projects').iterdir() if p.is_dir() and any(k in p.name.lower() for k in ['expo-ft-sz2','opendw-robotwin','robodojo','dojo-'])]
for root in roots:
 if not root.exists():continue
 out['roots'].append(str(root))
 for folder,ds,fs in os.walk(root,followlinks=False):
  p=Path(folder)
  try:
   st=p.stat();ident=(st.st_dev,st.st_ino)
   if st.st_uid!=uid or ident in seen:ds[:]=[];continue
   seen.add(ident)
  except OSError as e:out['errors'].append([str(p),type(e).__name__]);ds[:]=[];continue
  ds[:]=[d for d in ds if d not in skip and len(p.relative_to(root).parts)<13]
  for d in list(ds):
   if (p/d).is_symlink():out['links'].append([str(p/d),os.readlink(p/d)]);ds.remove(d)
  for n in fs:
   f=p/n
   try:
    st=f.stat()
    if st.st_uid!=uid or f.is_symlink():continue
    row={'path':str(f),'bytes':st.st_size,'mtime':st.st_mtime}
    if f.suffix in ['.pt','.pth','.safetensors','.bin'] and ('checkpoint' in str(f) or 'global_step_' in str(f)):
     if 'replay' not in n and 'optimizer' not in n:out['weights'].append(row)
    if (n.startswith('events.out.tfevents.') or f.suffix.lower() in ['.json','.jsonl','.yaml','.yml','.csv','.tsv','.log','.txt','.md','.out','.err']) and st.st_size<=20*1024**2:
     out['files'].append(row)
   except OSError as e:out['errors'].append([str(f),type(e).__name__])
out['finished']=time.time();print(json.dumps(out))
