import collections,datetime,hashlib,json,os,shutil,socket,subprocess,sys,time,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
R=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001');T=R/'formal-turn-switch-repair-20261002';C=R/'parallel-trial-20261006';S=C/'source'
CO=Path('/data/chenyiteng/deployment-20261006/rlt-q-signals-g67-v1/coexist')
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu02' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path.insert(0,str(S/'tools'))
from expo_smoke_owner import owned,identity
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'time':time.time(),'beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'host':socket.gethostname(),'read_only':True}
out['routing']={str(p):read(p) for p in [T/'active-continuation.json',C/'current.json',CO/'current.json',CO/'owner.json'] if p.is_file()}
out['status']=read(T/'run/status.json');out['checkpoint']=read(T/'run/checkpoint.json')
driver_id=identity(out['status']['pid']);out['driver_identity']=driver_id
out['driver_alive']=bool(driver_id and driver_id['uid']==20001 and driver_id['start_ticks']==410425820)
out['heartbeat_age_seconds']=time.time()-(T/'driver-heartbeat').stat().st_mtime
out['live_routing_identities']={}
for path,v in out['routing'].items():
 for key in ('owner','child','expo_driver'):
  if isinstance(v.get(key),dict) and 'pid' in v[key]:out['live_routing_identities'][path+'#'+key]={'identity':v[key],'alive':owned(v[key])}
out['error_files']={str(p):read(p) for p in [C/'final.json',C/'error.json',CO/'error.json',CO/'final.json',T/'run/failure.json'] if p.exists()}
inputs=read(T/'inputs.json');out['source_commit']=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
out['source_manifest']={p:{'expected':h,'actual':sha(S/p)} for p,h in inputs['port_source_manifest'].items()}
out['all_source_files_match']=all(v['expected']==v['actual'] for v in out['source_manifest'].values())
out['formal']=inputs['formal'];out['core']=inputs['core'];out['evaluation']=inputs['evaluation']
out['evaluations']={p.parent.name:read(p) for p in (T/'run/evaluations').glob('*/complete.json')}
out['eval_metadata_files']={str(p.relative_to(T/'run/evaluations')):read(p) for p in (T/'run/evaluations').glob('*/*.json') if p.stat().st_size<300000 and p.name!='complete.json'}
out['events']=[];counts=collections.Counter();eventfile=T/'run/events.jsonl'
with eventfile.open() as f:
 for line in f:
  try:row=json.loads(line)
  except json.JSONDecodeError:continue
  counts[row.get('event')]+=1
  if row.get('event') in ('resume_verified','episode_started','episode_finished','learner_finished','evaluation_finished','failure','failed','stopped','complete'):
   if row['event']=='episode_finished':row={k:v for k,v in row.items() if k!='replay_entry'}
   out['events'].append(row)
out['event_counts']=dict(counts)
out['checkpoints']={}
for p in (T/'run').glob('checkpoint-*.pt*'):
 d={'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'inode':p.stat().st_ino}
 if p.suffix=='.pt':
  try:
   with zipfile.ZipFile(p) as z:d.update(zip_directory_ok=True,zip_members=len(z.infolist()),uncompressed_bytes=sum(i.file_size for i in z.infolist()))
  except Exception as e:d['zip_directory_error']=repr(e)
 out['checkpoints'][p.name]=d
idx=read(T/'replay/index.json');out['replay_index_type']=type(idx).__name__
out['replay_index_keys']=list(idx)[:30] if isinstance(idx,dict) else None
out['replay_index']=idx
out['inventory']={};out['video_files']=[]
for label,folder in [('run',T/'run'),('replay',T/'replay'),('control',C),('coexist',CO)]:
 count=total=0;exts=collections.Counter();largest=[]
 for current,dirs,files in os.walk(folder):
  dirs[:]=[x for x in dirs if x not in ('.git','source','publication.git','__pycache__')]
  for name in files:
   p=Path(current)/name
   try:st=p.stat()
   except OSError:continue
   count+=1;total+=st.st_size;exts[p.suffix]+=1
   largest.append({'path':str(p),'bytes':st.st_size});largest=sorted(largest,key=lambda x:x['bytes'],reverse=True)[:8]
   if p.suffix.lower() in ('.mp4','.webm','.gif'):out['video_files'].append({'path':str(p),'bytes':st.st_size})
 out['inventory'][label]={'files':count,'bytes':total,'suffix_counts':dict(exts),'largest':largest}
out['control_files']={str(folder):[p.name for p in folder.iterdir()] for folder in [C,CO]}
out['queues']={str(p):read(p/'queue-status.json') for p in [Path('/data/chenyiteng/deployment-20261002/rlt-next6-'+task) for task in ('place_object_stand','move_playingcard_away')]}
xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=25))
out['gpus']=[{'index':i,'uuid':g.findtext('uuid'),'memory':g.findtext('fb_memory_usage/used'),'util':g.findtext('utilization/gpu_util'),'processes':[{k.tag:k.text for k in p} for p in g.findall('processes/process_info')]} for i,g in enumerate(xml.findall('gpu'))]
out['memory']={r.split(':')[0]:r.split(':')[1].strip() for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith(('MemTotal:','MemAvailable:'))}
out['process_memory']={}
if out['driver_alive']:
 out['process_memory']={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc',str(driver_id['pid']),'status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
out['disk']={p:dict(zip(('total','used','free'),shutil.disk_usage(p))) for p in ('/','/home',str(T/'run'))}
out['log_tails']={}
for p in [C/'formal.log',CO/'owner.log']:
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));out['log_tails'][str(p)]=f.read().decode(errors='replace')
print(json.dumps(out))
