set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import csv, datetime, json, os, re, subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1')
def read(p):
    try:return p.read_text(errors='replace').strip()
    except OSError:return None
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',read(run/'driver.log') or '')
lines=log.splitlines()
state={f:read(run/f) for f in ('wrapper.pid','started_at.txt','finished_at.txt','exit_code.txt')}
matches=re.findall(r'Global Step:\s*(\d+)\s*/',log)
errors=['CUDA out of memory','OutOfMemoryError','Fatal Python error','OIDN Error','pthread_key_create failed','RuntimeError:','ValueError:','Traceback','Exiting main process due to a failure']
result={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'state':state,'step':int(matches[-1]) if matches else 0,'errors':{e:log.count(e) for e in errors},'error_lines':[l for l in lines if any(e in l for e in errors)][:8],'tail':lines[-10:],'dvac_log':[l for l in lines if 'Online BC DVAC:' in l][-3:]}
if state['wrapper.pid']:
    p=Path('/proc')/state['wrapper.pid'];result['wrapper_alive']=p.exists()
result['workers']={g:int(pid) for g,pid in re.findall(r'((?:Actor|Env|Rollout)Group)\(rank=0\) pid=(\d+)',log)}
result['scalars']={}
if (run/'tensorboard').exists():
    ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
    for tag in ea.Tags().get('scalars',[]):
        if any(s in tag for s in ('success','loss','dvac','grad_norm','time/step')):
            result['scalars'][tag]=[[e.step,e.value] for e in ea.Scalars(tag)]
result['checkpoints']=[]
for root in (run/'checkpoints',run/'pi0-adjust-bottle-online-bc-dvac-smoke-gpu7/checkpoints'):
    if root.exists():
        for step in sorted(root.glob('global_step_*')):
            result['checkpoints'].append({'path':str(step),'files':[{'name':str(f.relative_to(step)),'bytes':f.stat().st_size} for f in step.rglob('*') if f.is_file()]})
if (run/'resource.csv').exists():
    rows=list(csv.DictReader((run/'resource.csv').open()))
    result['resources']={'last':rows[-1] if rows else None,'samples':len(rows),
        'gpu_peak_mib':max([float(x['gpu7_used_mib']) for x in rows if x['gpu7_used_mib']]+[0]),
        'ram_available_min_gib':min([float(x['host_mem_available_kib'])/1024**2 for x in rows if x['host_mem_available_kib']]+[1e9]),
        'env_fd_max':max([int(x['env_open_fds']) for x in rows if x['env_open_fds']]+[0])}
result['gpu']=subprocess.run(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
print(json.dumps(result,ensure_ascii=False))
PY
