set -eu
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0
nice -n 19 ionice -c 3 /usr/bin/python3 -B - <<'PY'
import collections,datetime,json,os,re,stat,subprocess,time
from pathlib import Path
base=Path('/data/chenyiteng/results');rows=[];skip=[];errors=[];checkpoints={};uid=os.getuid()
video={'.mp4','.avi','.mov','.mkv','.webm','.gif'}
for curr,dirs,files in os.walk(base,followlinks=False):
 dirs[:]=[d for d in dirs if d!='.git' and not (Path(curr)/d).is_symlink()]
 for name in files:
  p=Path(curr)/name
  try:
   s=p.lstat()
   if not stat.S_ISREG(s.st_mode):continue
   item={'path':str(p),'bytes':s.st_size,'allocated':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'nlink':s.st_nlink,'uid':s.st_uid}
   parts=p.parts
   reason='video' if p.suffix.lower() in video else 'large_50MiB' if s.st_size>=50*1024**2 else 'foreign_owner' if s.st_uid!=uid else None
   (skip if reason else rows).append(dict(item,reason=reason) if reason else item)
   match=re.search(r'^(.*?/checkpoints)/(global_step_(\d+))/',str(p))
   if match:
    cr,gen,step=match.groups();g=checkpoints.setdefault(cr,{}).setdefault(int(step),{'path':cr+'/'+gen,'files':[]});g['files'].append(item)
  except OSError as e:errors.append({'path':str(p),'error':str(e)})
open_files={};active=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=uid:continue
  command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if any(t in command for t in ['train_embodied','pi05','dsrl','rlt','wrapper.sh']):active.append({'pid':int(proc.name),'cmdline':command})
  for fd in (proc/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if target.startswith(str(base)+'/'):open_files.setdefault(target,[]).append(int(proc.name))
   except OSError:pass
 except OSError:pass
summary=collections.defaultdict(lambda:{'files':0,'bytes':0})
for r in rows:
 key=str(Path(r['path']).relative_to(base)).split('/')[0];summary[key]['files']+=1;summary[key]['bytes']+=r['bytes']
d={'time':datetime.datetime.now().astimezone().isoformat(),'root':str(base),'archive_candidates':rows,'excluded':skip,'archive_summary':dict(summary),'errors':errors,
'checkpoints':{k:[dict(step=s,**v) for s,v in sorted(gs.items())] for k,gs in checkpoints.items()},'open_files':open_files,'active_processes':active}
print(json.dumps(d,ensure_ascii=False))
PY
