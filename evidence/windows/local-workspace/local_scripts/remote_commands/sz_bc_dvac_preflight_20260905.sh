set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import datetime, hashlib, json, os, subprocess
from pathlib import Path
base=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
root=base/'pi0-online-bc'
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1')
def cmd(*args):
    p=subprocess.run(args,text=True,capture_output=True,timeout=20)
    return dict(rc=p.returncode,out=p.stdout.strip(),err=p.stderr.strip())
files=['rlinf/data/online_bc.py','rlinf/workers/actor/fsdp_online_bc_policy_worker.py','rlinf/models/embodiment/openpi/openpi_action_model.py','rlinf/workers/env/env_worker.py','rlinf/workers/rollout/hf/huggingface_worker.py','examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml','tests/unit_tests/test_online_bc.py']
lines=(run/'driver.log').read_text(errors='replace').splitlines()
errs=[i for i,l in enumerate(lines) if 'CUDA out of memory' in l or 'OutOfMemoryError' in l]
result=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),head=cmd('git','-C',str(root),'rev-parse','HEAD'),dirty=cmd('git','-C',str(root),'status','--porcelain'),branch=cmd('git','-C',str(root),'branch','--show-current'),hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},dvac_path_exists=(base/'pi0-online-bc-dvac').exists(),dvac_branch=cmd('git','-C',str(root),'show-ref','--verify','refs/heads/codex/sz-pi0-online-bc-dvac'),oom_context=['\n'.join(lines[max(0,i-65):i+16]) for i in errs],log_tail=lines[-30:],root_files=cmd('find',str(run),'-maxdepth','2','-type','f','-printf','%P %s\n'),gpu=cmd('nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'))
print(json.dumps(result,ensure_ascii=False))
PY
