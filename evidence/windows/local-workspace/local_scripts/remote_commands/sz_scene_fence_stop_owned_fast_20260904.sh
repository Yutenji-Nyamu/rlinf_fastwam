#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
date -Is
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import os, signal, time, subprocess, json, datetime
from pathlib import Path
import ray
run=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1')
rt=run/'runtime'
assert os.getuid()==1003
assert (rt/'wrapper.pid').read_text().strip()=='1052625'
assert (rt/'owned.pgid').read_text().strip()=='1052625'
protected=[321933,322685,3176215]
def proc(pid):
 p=Path('/proc')/str(pid)
 s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return {'uid':p.stat().st_uid,'state':s[0],'ppid':int(s[1]),'pgid':int(s[2]),'start':s[19],
         'cmd':(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')}
def env(pid):
 return dict(s.split('=',1) for s in (Path('/proc')/str(pid)/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in s)
before={pid:proc(pid) for pid in protected}
wrapper=proc(1052625)
assert wrapper['uid']==1003 and str(rt/'wrapper.sh') in wrapper['cmd'],wrapper
pids={int(s) for s in subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()}
expected={1053112,1053114,1053115,1053117,1053118,1053120}
assert pids==expected, pids
for pid in pids:
 e=env(pid)
 assert proc(pid)['uid']==1003 and e.get('CLUSTER_NAMESPACE')=='RLinf_1',(pid,e.get('CLUSTER_NAMESPACE'))
 assert e.get('ROBOTWIN_PATH','').endswith('/robotwin-clean-oidn-off-20260904'),pid
print('VERIFIED_OWNED_GPU_PIDS',sorted(pids),flush=True)
ray.init(address='172.17.0.1:6389',namespace='codex_fast_scene_fence_stop',logging_level='ERROR')
rows=[x['name'] for x in ray.util.list_named_actors(all_namespaces=True) if x['namespace']=='RLinf_1']
required={'ActorGroup:0','ActorGroup:1','EnvGroup:0','EnvGroup:1','RolloutGroup:0','RolloutGroup:1','Actor:0','Env:0','Rollout:0','CollectiveManager','DeviceLockManager','NodeManager','PortLockManager','WorkerManager'}
assert set(rows)==required|{s for s in rows if s.startswith('NodeProbe_')} and len(rows)==15,rows
print('VERIFIED_NAMESPACE',rows,flush=True)
allproc={}
for p in Path('/proc').glob('[0-9]*'):
 try:allproc[int(p.name)]=proc(int(p.name))
 except (OSError,ValueError):pass
owned={1052625,1052626}
while True:
 new=owned|{pid for pid,s in allproc.items() if s['ppid'] in owned}
 if new==owned:break
 owned=new
assert not set(protected)&owned
for pid in owned:assert allproc[pid]['uid']==1003,pid
print('STOP_DRIVER_TREE',sorted(owned),flush=True)
# Give the driver a normal termination opportunity; the wrapper can record rc.
os.kill(1052633,signal.SIGTERM)
time.sleep(3)
for name in sorted(rows,key=lambda n:n in {'CollectiveManager','DeviceLockManager','NodeManager','PortLockManager','WorkerManager'}):
 try:ray.kill(ray.get_actor(name,namespace='RLinf_1'),no_restart=True)
 except ValueError:pass
deadline=time.monotonic()+20
while time.monotonic()<deadline:
 remaining=[pid for pid in owned|pids if Path('/proc',str(pid)).exists()]
 if not remaining:break
 time.sleep(1)
for pid in owned|pids:
 try:
  current=proc(pid)
  old=allproc[pid]
  if current['state']!='Z' and current['start']==old['start']:
   assert current['uid']==1003
   print('OWNED_KILL_IF_STILL_ALIVE',pid,flush=True)
   os.kill(pid,signal.SIGKILL)
 except FileNotFoundError:pass
time.sleep(3)
left=[x for x in ray.util.list_named_actors(all_namespaces=True) if x['namespace']=='RLinf_1']
assert not left,left
for pid in protected:assert proc(pid)['start']==before[pid]['start'],pid
gpu_left=subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert not gpu_left,gpu_left
record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'user authorized scene-fence fix and fresh restart with same256-trajectory contract','last_complete_step':1,'namespace':'RLinf_1','gpu_pids':sorted(pids),'shared_ray_and_sidney_unchanged':True,'checkpoints':[]}
target=rt/'stopped_for_scene_fence_20260904.json'
assert not target.exists()
target.write_text(json.dumps(record,indent=2)+'\n')
print('STOP_COMPLETE',json.dumps(record),flush=True)
ray.shutdown()
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
date -Is
