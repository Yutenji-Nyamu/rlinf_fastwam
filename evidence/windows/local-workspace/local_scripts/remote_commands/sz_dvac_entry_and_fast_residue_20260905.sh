#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - <<'PY'
import datetime,hashlib,json,os,re,subprocess,urllib.request
from pathlib import Path
assert os.getuid()==1003
def read(p):
    try:return Path(p).read_text(errors='replace')
    except OSError:return None
def cmd(args):
    p=subprocess.run(args,text=True,capture_output=True,timeout=20)
    return {'rc':p.returncode,'out':p.stdout.strip(),'err':p.stderr[-600:]}
def git(tree,*args):return cmd(['git','-C',str(tree),*args])['out']
base=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
fast=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3')
rt=fast/'runtime'
f=rt/'driver.log';log=read(f) or '';lines=log.splitlines()
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
out={'time':now,'diagnostic_pids_excluded':[os.getpid(),os.getppid()],'fast_runtime':str(rt),'state':{n:read(rt/n) for n in ['wrapper.pid','driver.pid','timeout.pid','finished_at.txt','exit_code.txt']},'log_mtime':datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat(),'log_bytes':f.stat().st_size,'fatal':[x for x in lines if 'Fatal Python error' in x][:5],'log_tail':lines[-12:],'fast_process_matches':[],'our_gpu_processes':[],'source_locks':[]}
gpu=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory,process_name','--format=csv,noheader,nounits'])
our_gpu_ids=set()
for l in gpu['out'].splitlines():
    a=[x.strip() for x in l.split(',')]
    try:
        if (Path('/proc')/a[1]).stat().st_uid==os.getuid():our_gpu_ids.add(int(a[1]))
    except (OSError,ValueError,IndexError):pass
needles=[str(fast),'fastwam-current-grpo','fastwam-scene-fence','fastwam-grpo-control','fastwam-action-dvac']
for p in Path('/proc').glob('[0-9]*'):
    try:
        if p.stat().st_uid!=os.getuid() or int(p.name) in out['diagnostic_pids_excluded']:continue
        args=(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
        env=(p/'environ').read_bytes().decode(errors='replace').split('\0')
        safe_env=[x for x in env if x.partition('=')[0] in ['RLINF_CODE_WORKING_DIR','RLINF_CONFIG_PATH','RAY_NAMESPACE','CUDA_VISIBLE_DEVICES','RLINF_SCENE_FENCE_LIBRARY','PYTHONPATH']]
        cwd=os.readlink(p/'cwd')
        matches=[n for n in needles if n in args or n in cwd or any(n in e for e in env)]
        rec={'pid':int(p.name),'comm':(read(p/'comm') or '').strip(),'cwd':cwd,'env':safe_env}
        if matches:out['fast_process_matches'].append({**rec,'matched':matches,'cmd':args[:1000]})
        if int(p.name) in our_gpu_ids:out['our_gpu_processes'].append(rec)
    except (OSError,ValueError):pass
out['old_pid_checks']=cmd(['ps','-p','1568962,1568964,1568973','-o','pid,ppid,lstart,stat,comm'])
out['gpu']=cmd(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
specs={
'pi0-online-bc':['rlinf/data/online_bc.py','rlinf/models/embodiment/openpi/openpi_action_model.py','rlinf/workers/actor/fsdp_online_bc_policy_worker.py'],
'rlt-dvac-pure-single-gpu-7d07a421':['rlinf/algorithms/rlt/dvac_weighting.py','rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py'],
}
for name,files in specs.items():
    tree=base/name
    out['source_locks'].append({'tree':str(tree),'head':git(tree,'rev-parse','HEAD'),'dirty':git(tree,'status','--porcelain'),'files':[{'path':rel,'sha256':hashlib.sha256((tree/rel).read_bytes()).hexdigest()} for rel in files]})
try:
    with urllib.request.urlopen('http://127.0.0.1:8265/api/v0/actors?limit=1000&detail=true&timeout=10',timeout=13) as response:
        data=json.load(response)
    # Keep actor metadata, not arbitrary task logs or external account payloads.
    out['ray_actor_query']=data
except Exception as e:out['ray_actor_query_error']=str(e)
print(json.dumps(out,ensure_ascii=False,indent=2))
PY
