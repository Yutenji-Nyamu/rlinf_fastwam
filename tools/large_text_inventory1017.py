import os,json,time
from pathlib import Path
os.nice(15);uid=os.getuid()
roots=[Path('/data/chenyiteng/results'),Path('/home/chenyiteng/results')]
roots += [p for p in Path('/data/chenyiteng/projects').iterdir() if p.is_dir() and any(k in p.name.lower() for k in ['expo-ft-sz2','opendw-robotwin','robodojo','dojo-'])]
skip={'.git','__pycache__','replay','replay_buffer','replay_buffers','success_replay','online','dataset','datasets','videos','video','images','frames','wandb','media','data','assets','models','source','src','venv','.venv','site-packages','node_modules','third_party','cache','.cache','conda','envs','venvs','packages','pkgs','lib','build','dist','Assets'}
ext={'.log','.jsonl','.json','.csv','.tsv','.yaml','.yml','.txt','.out','.err'}
out={'time':time.time(),'files':[],'large':[],'errors':[],'roots':[str(p) for p in roots]};seen=set()
for root in roots:
 if not root.exists():continue
 for folder,ds,fs in os.walk(root,followlinks=False):
  p=Path(folder)
  try:
   st=p.stat();ident=(st.st_dev,st.st_ino)
   if st.st_uid!=uid or ident in seen:ds[:]=[];continue
   seen.add(ident)
  except OSError as e:out['errors'].append([str(p),type(e).__name__]);ds[:]=[];continue
  ds[:]=[d for d in ds if d not in skip and not (p/d).is_symlink() and len(p.relative_to(root).parts)<13]
  if '/projects/' in str(root) and p==root:
   if 'robodojo' in root.name:ds[:]=[d for d in ds if d in ['runs','RoboDojo']]
   if 'opendw' in root.name:ds[:]=[d for d in ds if d in ['runs','wm-cycle-20261009-v2','rynn-numeric-v1','rynn-success-v1','rlt45-return-20261007-v1','logs','reports']]
  if 'robodojo' in root.name and p==root/'RoboDojo':ds[:]=[d for d in ds if d=='eval_result']
  for n in fs:
   f=p/n
   if f.suffix.lower() not in ext and not n.startswith('events.out.tfevents.'):continue
   try:
    st=f.stat()
    if st.st_uid!=uid or f.is_symlink():continue
    if st.st_size<=20*1024**2:continue
    row={'path':str(f),'bytes':st.st_size,'mtime':st.st_mtime}
    out['files' if st.st_size<=20*1024**2 else 'large'].append(row)
   except OSError as e:out['errors'].append([str(f),type(e).__name__])
out['finished']=time.time();print(json.dumps(out))
