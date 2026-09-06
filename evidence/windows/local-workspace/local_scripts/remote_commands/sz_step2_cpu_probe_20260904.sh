#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date -Is
V=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
timeout 30s "$V/bin/python" - <<'PY'
import time,importlib.util,importlib.metadata,threading,faulthandler
from concurrent.futures import ThreadPoolExecutor
faulthandler.dump_traceback_later(20,exit=True)
for n in ['toppra','mplib','sapien','numpy']:
 print('VERSION',n,importlib.metadata.version(n),flush=True)
for n in ['qpoases','cvxpy']:
 print('SPEC',n,importlib.util.find_spec(n),flush=True)
from toppra.solverwrapper import available_solvers
print('SOLVERS',available_solvers(False),flush=True)
barrier=threading.Barrier(16)
def probe(i):
 barrier.wait()
 for j in range(256):
  result=available_solvers(False)
 return i,result
t=time.monotonic()
with ThreadPoolExecutor(max_workers=16) as pool: results=list(pool.map(probe,range(16)))
print('CPU_PROBE_PASS',len(results)*256,'calls',time.monotonic()-t,'sec',flush=True)
faulthandler.cancel_dump_traceback_later()
PY
printf 'CPU_PROBE_RC=%s\n' "$?"
python3 - <<'PY'
from pathlib import Path
import subprocess,json
sp='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/py-spy'
r=subprocess.run([sp,'dump','--pid','1053120','--nonblocking','--json'],capture_output=True,text=True,timeout=12)
for t in json.loads(r.stdout):
 print('GIL_THREAD',json.dumps({k:v for k,v in t.items() if k!='frames'}),'TOP',t.get('frames',[{}])[0].get('name') if t.get('frames') else None)
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
for kind,slug in [('fastwam-grpo','fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1'),('pi05-sidney','move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')]:
 p=base/kind/'runs'/slug/'runtime/resolved.yaml'
 print('PATH_CONFIG',p)
 for line in p.read_text().splitlines():
  if any(k in line for k in ['save_path:','video_base_dir:','ray_tracing_denoiser:','planner_backend:','total_num_envs:','rollout_epoch:','global_batch_size:','resume:']):print(line)
root=base/'fastwam-grpo/runs'
for run in sorted(root.iterdir()):
 p=run/'runtime/driver.log'
 if not p.is_file():continue
 s=p.read_text(errors='replace')
 terms=['pthread_key_create failed','Fatal Python error','CUDA out of memory','SubEnv 0 step error','TimeoutError','DeviceLockManager','ModuleNotFoundError: No module named']
 print('HISTORICAL_LOG',run.name,'bytes',len(s),'signals',{k:s.count(k) for k in terms if k in s})
PY
nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
date -Is
