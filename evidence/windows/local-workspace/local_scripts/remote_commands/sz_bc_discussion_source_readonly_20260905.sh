#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import datetime, hashlib, re, subprocess
from pathlib import Path
tree=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc')
root=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc')
print('TIME',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
print(subprocess.check_output(['id'],text=True).strip())
for args in [['rev-parse','HEAD'],['status','--short']]:
    print('GIT',args,subprocess.check_output(['git','-C',str(tree),*args],text=True).strip())
for rel in ['examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml','rlinf/workers/actor/fsdp_online_bc_policy_worker.py','rlinf/models/embodiment/openpi/openpi_action_model.py','rlinf/envs/robotwin/robotwin_env.py','rlinf/workers/env/env_worker.py']:
    p=tree/rel
    print('SHA256',rel,hashlib.sha256(p.read_bytes()).hexdigest())
for pid in ['321933','322685']:
    p=Path('/proc')/pid/'limits'
    print('RAY_LIMIT',pid,[l for l in p.read_text().splitlines() if 'Max open files' in l])
for n,needle in [(4,'AssertionError'),(5,'getSemaphoreFdKHR'),(6,'cannot create buffer')]:
    run=root/f'pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v{n}'
    print('RUN',n,'EXIT',(run/'exit_code.txt').read_text().strip(),'FINISH',(run/'finished_at.txt').read_text().strip())
    lines=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace')).splitlines()
    matches=[i for i,l in enumerate(lines) if needle in l]
    for i in matches[:1]:
        print('\n'.join(lines[max(0,i-18):i+2]))
official=subprocess.check_output(['git','-C',str(tree),'show','dc9b87cc49334c7516487ead68ebeb060fd7c090:examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml'],text=True)
print('OFFICIAL_DAGGER_FIELDS')
for line in official.splitlines():
    if any(k in line for k in ['precision','use_orig_params','train_expert_only','sharding_strategy','update_epoch','batch_size','lr:']): print(line)
from openpi.models_pytorch import pi0_pytorch
import inspect
print('NATIVE_SOURCE',inspect.getsourcefile(pi0_pytorch.PI0Pytorch))
print(inspect.getsource(pi0_pytorch.PI0Pytorch.sample_noise))
print(inspect.getsource(pi0_pytorch.PI0Pytorch.sample_time))
PY
