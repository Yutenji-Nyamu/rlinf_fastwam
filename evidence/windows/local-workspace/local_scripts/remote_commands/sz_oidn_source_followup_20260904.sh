#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import subprocess,re,json
rt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6')
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo')
files=[(rt/'envs/_base_task.py',r'def (close_env|setup_scene)|denois'),(wt/'rlinf/envs/robotwin/robotwin_env.py',r'def (reset|offload|onload|_init|__init__|_handle_auto_reset)|ROBOTWIN_PATH|VectorEnv'),(wt/'rlinf/workers/env/env_worker.py',r'def (interact|evaluate)|\.offload\(|\.onload\(')]
for p,pat in files:
 lines=p.read_text().splitlines();sel=set()
 for i,l in enumerate(lines):
  if re.search(pat,l):sel.update(range(max(0,i-4),min(len(lines),i+85)))
 print('SOURCE',str(p));print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(sel)))
ray=Path('/data/chenyiteng/ray/rlt-dsrl-v3/session_2026-08-23_16-27-53_911161_321906/logs')
for pid in (3590591,2813140,2813142):
 for parent in (ray,ray/'old'):
  for p in parent.glob(f'worker-*-{pid}.err'):
   lines=p.read_text(errors='replace').splitlines();sel=set()
   for i,l in enumerate(lines):
    if re.search(r'(?:_base_task|vector_env|robotwin_env|env_worker)\.py|Current thread|Fatal Python error',l):sel.update(range(max(0,i-2),min(len(lines),i+4)))
   print('STACK_CONCISE',str(p));print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(sel)))
venv=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages')
for pat in ('sapien-*.dist-info/METADATA','sapien-*.dist-info/direct_url.json'):
 for p in venv.glob(pat):print('METADATA',str(p),'\n'.join(p.read_text().splitlines()[:15]))
for p in (venv/'sapien').rglob('*'):
 if p.is_file() and any(k in p.name.lower() for k in ('oidn','openimagedenoise','svulkan','libtbb')):print('NATIVE_LIB',str(p),p.stat().st_size)
for pid in (3177205,3177207):
 p=Path('/proc',str(pid));print('SIDNEY_ENV_PROC',pid)
 if not p.exists(): continue
 status=p.joinpath('status').read_text(); print('\n'.join(l for l in status.splitlines() if l.startswith(('VmRSS','VmSize','Threads','State'))))
 env=dict(x.split('=',1) for x in p.joinpath('environ').read_text().split('\0') if '=' in x)
 print(json.dumps({k:env[k] for k in ('ROBOTWIN_PATH','PYTHONPATH','RLINF_CODE_WORKING_DIR','CUDA_VISIBLE_DEVICES') if k in env}))
 print('CGROUP',p.joinpath('cgroup').read_text())
 print('LIMITS',p.joinpath('limits').read_text())
print('NETWORK_HEADERS')
for label,proxy in [('direct',''),('proxy','http://127.0.0.1:7890')]:
 for url in ('https://github.com','https://huggingface.co'):
  args=['curl','--head','--silent','--show-error','--output','/dev/null','--max-time','12','--connect-timeout','5','--write-out','%{http_code} %{time_total} %{remote_ip}', '--noproxy','*'] if not proxy else ['curl','--head','--silent','--show-error','--output','/dev/null','--max-time','12','--connect-timeout','5','--write-out','%{http_code} %{time_total} %{remote_ip}','--proxy',proxy,'--noproxy','']
  r=subprocess.run(args+[url],capture_output=True,text=True,timeout=15);print(label,url,'rc',r.returncode,r.stdout,r.stderr)
PY
