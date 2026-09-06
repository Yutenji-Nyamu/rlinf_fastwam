set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import datetime,json,os,re,subprocess
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1')
root='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-dvac'
base=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc')
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace'))
workers=dict(re.findall(r'((?:Actor|Env|Rollout)Group)\(rank=0\) pid=(\d+)',log))
whitelist={'CUDA_VISIBLE_DEVICES','REPO_PATH','RLINF_CODE_WORKING_DIR','PI05_MODEL_PATH','ONLINE_BC_RUN_DIR','LD_PRELOAD','RLINF_SCENE_FENCE_LIBRARY'}
states={}
for role,pid in workers.items():
 proc=Path('/proc')/pid
 assert proc.stat().st_uid==os.getuid()
 env=dict(s.split('=',1) for s in (proc/'environ').read_text().split('\0') if '=' in s)
 states[role]={'pid':pid,'env':{k:v for k,v in env.items() if k in whitelist}}
 assert env['CUDA_VISIBLE_DEVICES']=='7' and env['REPO_PATH']==env['RLINF_CODE_WORKING_DIR']==root
 assert env['ONLINE_BC_RUN_DIR']==str(run) and not env.get('LD_PRELOAD') and not env.get('RLINF_SCENE_FENCE_LIBRARY')
children=subprocess.check_output(['ps','--ppid',(run/'timeout.pid').read_text().strip(),'-o','pid=,args='],text=True).strip()
assert '+online_bc_model=pi05_sidney +bc_dvac=bounded_half' in children and 'runner.save_interval=10' in children
errors=[s for s in log.splitlines() if re.search(r'Fatal Python|Traceback \(most recent|OutOfMemoryError|CUDA out of memory|ErrorInitializationFailed|cannot create buffer|AssertionError|RuntimeError:',s)]
out={'time':datetime.datetime.now().astimezone().isoformat(),'driver':children,'workers':states,'source_head':(run/'runtime/source-head.txt').read_text().strip(),'rollout_started':'Generating Rollout Epochs' in log,'errors':errors[:5],'tail':log.splitlines()[-9:],'bc_base_head':subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip(),'bc_base_dirty':subprocess.check_output(['git','-C',str(base),'status','--porcelain'],text=True).strip(),'protected_pids':{str(p):Path('/proc',str(p)).exists() for p in (2143105,321933,322685,602620)}}
print(json.dumps(out))
PY
