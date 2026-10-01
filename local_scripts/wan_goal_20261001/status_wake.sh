set -eu
/data/chenyiteng/projects/wan-goal-sz3/envs/pi05-wan/bin/python -B - <<'PY'
import json,subprocess,sys,time
from pathlib import Path
sys.modules.setdefault('tensorflow',None)
R=Path('/data/chenyiteng/projects/wan-goal-sz3')
P=Path('/data/chenyiteng/projects/robodojo-openwam-sz3')
D=P/'runs/sz3_pi05_official_6300_n4_dual_20260929_r2'
W=R/'runs/wan-goal-sz3-20261001-r3'
sys.path.insert(0,str(R/'scripts/resource_switch'))
from common import account,alive
account()
def read(p): return json.loads(p.read_text()) if p.is_file() else None
active=read(D/'active-continuation.json')
report={'time':time.time(),'active_attempt':active['attempt_dir'],'owner_alive':alive(active),
 'pipeline':read(D/'pipeline-current.json'),'sequence':read(W/'sequence-current.json'),'stages':{}}
for name in ('pi05-smoke','pi05-formal'):
 stage=W/name;control=W/(name+'-control')
 if not stage.exists(): continue
 row={'launch':read(stage/'launch.json'),'placement_verified':(stage/'verified-placement.json').is_file(),
      'exit':read(stage/'wm-exit.json'),'verification':read(control/'smoke-verification.json')}
 log=stage/'command.log'
 if log.is_file():
  with log.open('rb') as stream:
   stream.seek(max(0,log.stat().st_size-10000));data=stream.read().decode('utf-8',errors='replace')
  row.update(log_age_seconds=time.time()-log.stat().st_mtime,log_tail=data[-6500:])
 tb=stage/'tensorboard/all'
 if list(tb.glob('events.out.tfevents.*')):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  acc=EventAccumulator(str(tb),size_guidance={'scalars':0});acc.Reload()
  row['metrics']={k:[{'step':v.step,'value':v.value,'wall_time':v.wall_time} for v in acc.Scalars(k)[-2:]]
   for k in acc.Tags().get('scalars',[]) if any(x in k for x in ('grad_norm','total_loss',
    'loss_mask_fraction','advantages_','success','episode_return','time/total'))}
  row['scalar_tags']=acc.Tags().get('scalars',[])
 report['stages'][name]=row
for label,argv in [('gpu',['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits']),
 ('compute',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),
 ('disk',['df','-B1','/','/data'])]:
 result=subprocess.run(argv,capture_output=True,text=True,timeout=20);report[label]={'exit':result.returncode,'out':result.stdout}
print(json.dumps(report))
PY
