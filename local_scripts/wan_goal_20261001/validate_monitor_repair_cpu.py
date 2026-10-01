"""CPU-only reviewed source checks; no workload or GPU launch."""
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import time

R=Path('/data/chenyiteng/projects/wan-goal-sz3')
sys.path.insert(0,str(R/'scripts/resource_switch'))
from common import account,sha
account()
files=['resource_switch/'+p.name for p in (R/'scripts/resource_switch').glob('*.py')]
files+=['wm_sequence.py','private_ray_driver.py','prepare_monitor_repair.py','check_direct_return_cpu.py']
directory=R/'tmp/monitor-repair-pycache-20261001'
directory.mkdir(mode=0o700)
for relative in files:
    py_compile.compile(str(R/'scripts'/relative),cfile=str(directory/(relative.replace('/','_')+'c')),doraise=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
tests=[]
for relative in ['resource_switch/check_resource_switch_cpu.py','check_direct_return_cpu.py']:
    started=time.monotonic()
    result=subprocess.run([sys.executable,'-B',str(R/'scripts'/relative)],capture_output=True,text=True,env=env,timeout=120)
    tests.append(dict(source=relative,exit_code=result.returncode,seconds=time.monotonic()-started,
                      stdout=result.stdout,stderr=result.stderr))
receipt=dict(time=time.time(),ok=all(t['exit_code']==0 for t in tests),cuda_initialized=False,
    scope='CPU identity and fixture control flow only; no actual resource release or training',
    source_sha256={relative:sha(R/'scripts'/relative) for relative in files},tests=tests,
    repaired_stress_receipt_sha256=sha(R/'logs/process-identity-repaired-20261001.json'),
    baseline_stress_receipt_sha256=sha(R/'logs/process-identity-baseline-20261001.json'),
    inode_receipt_sha256=sha(R/'logs/proc-inode-evidence-20261001.json'))
target=R/'logs/monitor-repair-cpu-20261001.json'
with target.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt['ok'] else 1)
