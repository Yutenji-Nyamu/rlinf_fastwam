"""Launch the single reviewed fresh-start owner, once, after CPU preparation."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
F = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-two-gpu-from0-v2')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ready = read(F / 'ready.json')
assert ready['cpu_validate_passed'] and ready['resume_step'] == 0
assert ready['plan_sha256'] == sha(ready['plan'])
plan = read(ready['plan'])
assert not Path(plan['owner_dir']).exists() and not (F / 'launch.json').exists()
spec = importlib.util.spec_from_file_location('fresh_launch_check', ready['entrypoint'])
wrapper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wrapper
spec.loader.exec_module(wrapper)
wrapper.install(plan).validate(plan)
argv = [plan['python'], '-u', '-B', ready['entrypoint'], '--plan', ready['plan'], 'owner']
with (F / 'owner-console.log').open('x') as stream:
    child = subprocess.Popen(argv, cwd=plan['repo'],
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1'),
                             stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
proc = Path('/proc') / str(child.pid)
stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
assert proc.stat().st_uid == 20001
receipt = {'time': datetime.datetime.now().astimezone().isoformat(), 'pid': child.pid, 'uid': 20001,
           'start': int(stat[19]), 'argv': argv, 'plan_sha256': sha(ready['plan']), 'physical_gpus': [4, 5],
           'start_policy': 'original_pi05', 'resume_step': 0, 'max_steps': 200}
(F / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
