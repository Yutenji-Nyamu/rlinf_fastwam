"""Bounded read-only GPU layout and phase memory, without model inference."""
import datetime,hashlib,json,os,socket,statistics,subprocess,time
from collections import defaultdict
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');O=S/'runs/lift-pot-v1'
out={'time':datetime.datetime.now().astimezone().isoformat(),'resources':{},'services':{}}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
mapping={row.split(',')[1].strip():int(row.split(',')[0]) for row in gpu.splitlines()}
out['gpu']=gpu
out['contexts']=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','m'],text=True,timeout=20)
def tailrows(path,limit):
 with path.open('rb') as f:
  offset=max(0,path.stat().st_size-limit);f.seek(offset)
  if offset:f.readline()
  for line in f:
   try:yield json.loads(line)
   except (ValueError,UnicodeDecodeError):continue
memory=defaultdict(list);perpid={};count=0;first=None;last=None
path=O/'resources.jsonl';cutoff=time.time()-100*60
for row in tailrows(path,64*1024*1024):
 if row.get('phase')!='formal':continue
 timestamp=datetime.datetime.fromisoformat(row['time']).timestamp()
 if timestamp<cutoff:continue
 count+=1;first=first or row['time'];last=row['time'];totals=defaultdict(float)
 owned={r['pid'] for r in row.get('processes',[])}
 for line in row.get('compute_memory_csv','').splitlines():
  parts=[x.strip() for x in line.split(',')]
  if len(parts)!=3 or parts[1] not in mapping:continue
  try:pid=int(parts[0]);value=float(parts[2].split()[0])
  except ValueError:continue
  if pid not in owned:continue
  g=mapping[parts[1]];totals[g]+=value
  if value>perpid.get(str(pid),{}).get('peak_mib',-1):perpid[str(pid)]={'gpu':g,'peak_mib':value,'time':row['time']}
 for g in range(4,8):memory[g].append(totals.get(g,0))
out['resources']={'file_bytes':path.stat().st_size,'count':count,'first':first,'last':last,'owned_compute_mib':{g:{'min':min(v),'median':statistics.median(v),'max':max(v)} for g,v in memory.items()},'per_pid_peak':perpid,'scope':'last 100 minutes, owner formal phase, sampled not exact allocator peak'}
for key in ['wm6','wm7']:
 p=O/'services'/key/'records/service-events.jsonl'
 rows=list(tailrows(p,12*1024*1024))
 batches=[r for r in rows if r.get('event')=='batch_completed' and r.get('actual_wm_batch')==16]
 selected=[]
 for r in batches[-6:]:selected.append({k:r.get(k) for k in ['timestamp_utc','seconds','world_model_seconds','reward_seconds','actual_wm_batch','wm_peak','reward_peak','cuda_allocated_bytes','cuda_reserved_bytes']})
 onload=[r for r in rows if r.get('event')=='onload_completed']
 out['services'][key]={'recent_full_batches':selected,'loaded_idle':[{k:r.get(k) for k in ['timestamp_utc','cuda_allocated_bytes','cuda_reserved_bytes']} for r in onload[-2:]],'file_bytes':p.stat().st_size}
cfg=json.loads((S/'lift-pot-v1/prepared/formal.yaml').read_text())
out['config']={k:cfg[k] for k in ['cluster']}
out['config']['actor']={k:v for k,v in cfg['actor'].items() if k in ['micro_batch_size','global_batch_size','enable_offload','fsdp_config','strategy','model_type','gradient_checkpointing']}
out['config']['rollout']={k:v for k,v in cfg['rollout'].items() if k in ['enable_offload','pipeline_stage_num','micro_batch_size','batch_size']}
print(json.dumps(out))
