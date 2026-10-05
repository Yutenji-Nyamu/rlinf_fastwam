"""Launch the one reviewed normal-reborrow v2 owner, never replay V1 handoff."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D = S / 'rynn-control-v2'
OWNER = S / 'runs/rynn-success-v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert not OWNER.exists() and not (D / 'launch-receipt.json').exists()
    prepared = read(D / 'prepared.json')
    path = D / 'prepared/plan.json'
    assert prepared['cpu_preflight_passed'] is True and prepared['plan_sha256'] == sha(path)
    plan = read(path)
    script = D / 'code/rynn_formal_owner_v2.py'
    assert plan['source_sha256'][str(script)] == sha(script)
    spec = importlib.util.spec_from_file_location('rynn_v2_launch_owner', script)
    owner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = owner
    spec.loader.exec_module(owner)
    module = owner.install(plan)
    module.validate(plan)
    cycle = Path(plan['lifecycle_path'])
    assert not (cycle / 'rlt-stopped.json').exists() and not (cycle / 'return-started.json').exists()
    identities = {key: row['original_identity'] for key, row in module.C.load_plan(cycle)['runs'].items()}
    assert set(identities) == {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
    assert all(module.H.same(value) for value in identities.values()), 'Prepared RLT driver changed; do not launch stale plan'
    before = module.H.gpu_processes(list(range(8)))
    outside = [row for row in before if row['gpu'] not in (4, 5, 6, 7)
               and (module.H.proc(row['pid']) or {}).get('uid') == 20001]
    assert not outside, 'Owned compute/graphics residue outside GPU4-7 needs transition audit'
    argv = [plan['python'], '-u', '-B', str(script), '--plan', str(path), 'owner']
    with (D / 'owner-launch.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=plan['repo'], env=read(plan['environment_file']),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = module.H.proc(child.pid)
    assert identity and identity['uid'] == 20001
    receipt = dict(time=datetime.datetime.now().astimezone().isoformat(), identity=identity, argv=argv,
        plan=str(path), plan_sha256=sha(path), rlt_before=identities, gpu_before=before,
        restart_kind='fresh normal reborrow; V1 held-resource handoff not replayed')
    with (D / 'launch-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
