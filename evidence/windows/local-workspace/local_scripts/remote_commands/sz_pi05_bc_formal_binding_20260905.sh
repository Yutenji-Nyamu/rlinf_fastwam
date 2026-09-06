set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import datetime,json,os,re,subprocess
from pathlib import Path
import yaml
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1')
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace'))
expected=yaml.safe_load((run/'runtime/resolved.yaml').read_text())
actual=None
for m in re.finditer(r'^\{',log,re.M):
 try:candidate,_=json.JSONDecoder().raw_decode(log[m.start():])
 except json.JSONDecodeError:continue
 if isinstance(candidate,dict) and 'runner' in candidate and 'actor' in candidate:
  actual=candidate;break
workers=dict(re.findall(r'((?:Actor|Env|Rollout)Group)\(rank=0\) pid=(\d+)',log))
allowed={'CUDA_VISIBLE_DEVICES','RLINF_CODE_WORKING_DIR','REPO_PATH','PI05_MODEL_PATH','ONLINE_BC_RUN_DIR','RAY_ADDRESS','LD_PRELOAD','RLINF_SCENE_FENCE_LIBRARY'}
states={}
for group,pid in workers.items():
 proc=Path('/proc')/pid
 assert proc.stat().st_uid==os.getuid()
 environ=dict(s.split('=',1) for s in (proc/'environ').read_text().split('\0') if '=' in s)
 states[group]={'pid':pid,'environ':{k:v for k,v in environ.items() if k in allowed},'fd_count':len(list((proc/'fd').iterdir()))}
 assert environ.get('CUDA_VISIBLE_DEVICES')=='6',(group,states[group])
 assert environ.get('ONLINE_BC_RUN_DIR')==str(run),(group,states[group])
 assert not environ.get('LD_PRELOAD') and not environ.get('RLINF_SCENE_FENCE_LIBRARY')
timeout_pid=(run/'timeout.pid').read_text().strip()
children=subprocess.check_output(['ps','--ppid',timeout_pid,'-o','pid=,args='],text=True).strip()
assert 'runner.save_interval=10' in children and 'runner.max_epochs=100' in children
if actual is not None:assert actual==expected,'Actual driver config differs from validated resolved'
sidney=run.parents[1]/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1/runtime-resume100-to200'
out={'time':datetime.datetime.now().astimezone().isoformat(),'driver':children,'worker_bindings':states,'actual_config_matches':actual==expected if actual is not None else None,'driver_prefix_if_unparsed':log.splitlines()[:12] if actual is None else [],'namespace_lines':[s for s in log.splitlines() if 'namespace' in s.lower()][-5:],'rollout_started':'Generating Rollout Epochs' in log,'shared_ray_pids_present':{str(p):Path('/proc',str(p)).exists() for p in (321933,322685)},'sidney_wrapper_602620_present':Path('/proc/602620').exists()}
print(json.dumps(out))
PY
