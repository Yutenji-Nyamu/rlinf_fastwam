"""Read-only exact SZ3 WMRL handoff and data inventory; no GPU initialization."""
import datetime, hashlib, json, os, socket, subprocess
from pathlib import Path
assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
O=S/'runs/formal-b16-v1'
out={'time':datetime.datetime.now().astimezone().isoformat(),'json':{},'source':{},'checkpoints':{},'media':{}}
for p in [O/'owner-plan.json',O/'owner-identity.json',O/'process-catalog.json',O/'state.json',O/'formal/driver-identity.json',O/'formal/driver-started.json',O/'error.json',O/'final.json']:
    if p.is_file():out['json'][str(p)]=json.loads(p.read_text())
plan=out['json'][str(O/'owner-plan.json')]
for name in ['lifecycle_path']:
    path=Path(plan[name])
    for filename in ['plan.json','borrowed.json','adopted.json','returned.json','return-dispatched.json']:
        p=path/filename
        if p.is_file():out['json'][str(p)]=json.loads(p.read_text())
for trial in plan['trials']:
    p=Path(trial['config']);out['json'][str(p)]=json.loads(p.read_text())
repo=Path(plan['repo'])
srcs=[Path(plan['base_owner_module']),S/'formal-b16-control-v1/code/batch16_formal_owner.py',
      repo/'rlinf/envs/world_model/opendw_robotwin_env.py',repo/'rlinf/envs/world_model/opendw_adapter.py',
      repo/'rlinf/workers/actor/embodied_fsdp_actor_worker.py',repo/'rlinf/runners/embodied_runner.py']
for p in srcs:
    if p.is_file():out['source'][str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
for root,dirs,files in os.walk(O/'formal'):
    dirs[:]=[d for d in dirs if d not in ['videos','tensorboard','ray']]
    for name in files:
        p=Path(root)/name
        if 'checkpoints' in p.parts:out['checkpoints'][str(p)]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
        if name in ['driver.log']:out['driver_tail']=p.read_text(errors='replace')[-6000:]
for root,dirs,files in os.walk(O/'formal'):
    dirs[:]=[d for d in dirs if d not in ['checkpoints','ray','tensorboard']]
    for name in files:
        p=Path(root)/name
        if p.suffix.lower() in ['.mp4','.gif','.json'] and len(out['media'])<40:
            out['media'][str(p)]={'bytes':p.stat().st_size}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
out['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
out['repo_status']=subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True)
out['repo_head']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
out['reset_metadata']=json.loads((S/'data/adjust-bottle-clean50-reset.json').read_text())
print(json.dumps(out,ensure_ascii=False))
