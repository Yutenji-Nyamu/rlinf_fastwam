#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date -Is
SP=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/py-spy
timeout 20s "$SP" dump --pid 1053120 --native
python3 - <<'PY'
from pathlib import Path
import subprocess
root=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
for p in (root/'robotwin-clean-oidn-off-20260904').rglob('vector_env.py'):
 print('VECTOR_FILE',p)
 for i,line in enumerate(p.read_text().splitlines(),1):print(f'{i}: {line}')
old=root/'robotwin-vector-render-lifecycle-fix-0008ae6'
print('OLD_PATCH',subprocess.run(['git','-C',str(old),'diff','0008ae6800df9f75fc8de7098bacb01735fd8fd2','HEAD','--','robotwin/envs/vector_env.py'],capture_output=True,text=True).stdout)
for pid in [1052633,1053118,1053120,3176215,3177205,3177207]:
 p=Path('/proc')/str(pid)
 env={}
 for item in (p/'environ').read_bytes().decode(errors='replace').split('\0'):
  if '=' in item:
   k,v=item.split('=',1)
   if k in ['RAY_ADDRESS','CLUSTER_NAMESPACE','RLINF_CODE_WORKING_DIR','ROBOTWIN_PATH','PYTHONPATH','CUDA_VISIBLE_DEVICES']:env[k]=v
 print('PROCESS_SCOPE',pid,env)
print('RAY_NAMED_ACTORS')
PY
timeout 25s /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import ray,json
ray.init(address='172.17.0.1:6389',namespace='readonly-step2-diag-20260904',log_to_driver=False)
print(json.dumps(ray.util.list_named_actors(all_namespaces=True),sort_keys=True))
ray.shutdown()
PY
date -Is
