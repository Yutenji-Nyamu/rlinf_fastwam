set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import csv,datetime,json,re,subprocess
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1')
def cmd(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=15);return p.stdout.strip()
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace')) if (run/'driver.log').exists() else ''
lines=log.splitlines()
errors=[s for s in lines if re.search(r'Fatal Python|Traceback \(most recent|OutOfMemoryError|CUDA out of memory|ErrorInitializationFailed|cannot create buffer|pthread_key_create|AssertionError|RuntimeError:',s)]
state={name:(run/name).read_text().strip() if (run/name).is_file() else None for name in ('wrapper.pid','started_at.txt','exit_code.txt','finished_at.txt')}
if state['wrapper.pid']:state['wrapper_alive']=Path('/proc/'+state['wrapper.pid']).exists()
rows=list(csv.DictReader((run/'resource.csv').open())) if (run/'resource.csv').exists() else []
valid=[r for r in rows if r['gpu6_used_mib'].isdigit()]
workers=dict(re.findall(r'((?:Actor|Env|Rollout)Group)\(rank=0\) pid=(\d+)',log))
worker_states={}
for name,pid in workers.items():
    proc=Path('/proc')/pid
    worker_states[name]={'pid':pid,'present':proc.exists()}
    try:
        worker_states[name]['status']=[line for line in (proc/'status').read_text().splitlines() if line.startswith(('Name:','State:','Uid:'))]
    except FileNotFoundError:worker_states[name]['present']=False
root='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc'
out={'time':datetime.datetime.now().astimezone().isoformat(),'state':state,'errors':errors[:8],
 'latest_step_mentions':re.findall(r'Global Step[^\n]*',log)[-3:],
 'workers':workers,'worker_states':worker_states,
 'source_head':(run/'runtime/source-head.txt').read_text().strip(),
 'git_head':cmd(['git','-C',root,'rev-parse','HEAD']),
 'git_status':cmd(['git','-C',root,'status','--porcelain']),
 'gpu':cmd(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
 'memory':{s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemAvailable:','SwapFree:'))},
 'disk':cmd(['df','-B1','/data','/home']),
 'peak_gpu6_mib':max([int(r['gpu6_used_mib']) for r in valid],default=0),
 'max_env_fds':max([int(r['env_open_fds']) for r in rows if r['env_open_fds'].isdigit()],default=0),
 'last_resource':rows[-1] if rows else None,'tail':lines[-22:]}
print(json.dumps(out))
PY
