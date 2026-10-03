"""One-shot rebind of two already-complete Stage1 waiting queues to new EXPO return.

Never relaunches Stage1 or directly touches a training/Ray process. The original
waiting owners must already be identity-pinned SIGSTOP-held by the recovery.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TRAIN = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002')
ROOT = TRAIN.parent / 'eval10-continuation-20261003'
CYCLE = ROOT / 'rlt-cycle-expo-eval10-20261003-v1'
PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
sys.path.insert(0, str(TRAIN / 'source/tools'))
from expo_smoke_owner import identity, owned, atomic
from expo_process import pidfd_open, pidfd_send


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_signal(row, sig):
    assert owned(row)
    fd = pidfd_open(row['pid'])
    try:
        assert owned(row)
        pidfd_send(fd, sig)
    finally:
        os.close(fd)


def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    route = read(ROOT / 'current.json')
    assert route['status'] == 'PREPARED_WAITING_QUEUE_REBIND' and owned(route['owner'])
    assert not any((ROOT / n).exists() for n in ['queue-rebind-ready.json', 'queue-rebind-intent.json', 'stop-intent.json'])
    cycle = read(CYCLE / 'plan.json'); cycle_sha = sha(CYCLE / 'plan.json')
    assert cycle['cycle_id'] == CYCLE.name
    held = read(TRAIN / 'eval10-transition-20261003/queues-held.json')
    assert held['ok'] and len(held['queues']) == 2
    backups = ROOT / 'queue-backups'; backups.mkdir(mode=0o700, exist_ok=False)
    records = []
    for entry in held['queues']:
        stage = Path(entry['stage']); old_owner = entry['identity']
        assert owned(old_owner) and identity(old_owner['pid'])['state'] in ('T', 't')
        status = read(stage / 'queue-status.json')
        assert status['stage1'] == 'COMPLETE' and set(status['roles'].values()) == {'WAITING_BORROW_RETURN'}
        assert not any((stage / (role + suffix)).exists() for role in ('clean', 'combo')
                       for suffix in ('-cutover-bound.json', '-dispatched.json', '-failure.json'))
        original = read(stage / 'plan.json'); new = copy.deepcopy(original)
        for role in ('clean', 'combo'):
            gpu, = new['runs'][role]['gpus']
            new['gates'][role] = dict(kind='global', final=str(ROOT / 'final.json'),
                owner=dict(pid=route['owner']['pid'], uid=20001, start=route['owner']['start_ticks']),
                receipts=[str(ROOT / 'rlt-status.json')], gpu=gpu,
                cycle_plan=str(CYCLE / 'plan.json'), cycle_sha256=cycle_sha, anchors={})
        for target in new['old_runs']:
            gpu, = target['gpus']; row = cycle['runs']['gpu' + str(gpu)]
            target.update(run=row['new_run'], namespace=row['namespace'], identity=None)
        a = copy.deepcopy(original); b = copy.deepcopy(new)
        for key in ('gates', 'old_runs'):a.pop(key); b.pop(key)
        assert a == b, 'Training parameters or source changed during queue rebind'
        task = entry['task']
        for name, filename in [('plan.json', task + '-plan.json'), ('owner-identity.json', task + '-owner-identity.json'),
                               ('queue-status.json', task + '-queue-status.json')]:
            with (backups / filename).open('xb') as f:f.write((stage / name).read_bytes())
        saved_owner = read(backups / (task + '-owner-identity.json'))
        assert saved_owner['pid'] == old_owner['pid'] and saved_owner['start'] == old_owner['start_ticks']
        records.append(dict(task=task, stage=str(stage), old_owner=old_owner, original=original, new=new))
    atomic(ROOT / 'queue-rebind-intent.json', dict(time=time.time(), owner=route['owner'], queues=records))
    results = []
    for entry in records:
        stage = Path(entry['stage']); old_owner = entry['old_owner']; task = entry['task']
        assert read(stage / 'plan.json') == entry['original']
        assert owned(old_owner) and identity(old_owner['pid'])['state'] in ('T', 't')
        # These two owners had finished Stage1 and had never dispatched a role.
        checked_signal(old_owner, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while owned(old_owner) and time.monotonic() < deadline:time.sleep(.05)
        assert not owned(old_owner)
        atomic(stage / 'plan.json', entry['new'])
        command = [PY, '-u', '-B', str(ROOT / 'tools/queue_continuation.py'), '--stage', str(stage),
            '--previous-plan', str(backups / (task + '-plan.json')),
            '--old-owner-receipt', str(backups / (task + '-owner-identity.json'))]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
        with (stage / 'queue-continuation.log').open('xb') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        anchor = identity(process.pid)
        atomic(ROOT / (task + '-queue-launched.json'), dict(time=time.time(), owner=anchor, command=command))
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            assert process.poll() is None and owned(anchor), 'Continuation queue exited during registration'
            path = stage / 'owner-identity-continuation.json'
            if path.is_file():
                receipt = read(path)
                assert receipt['pid'] == anchor['pid'] and receipt['start'] == anchor['start_ticks']
                assert receipt['plan_sha256'] == sha(stage / 'plan.json')
                break
            time.sleep(.5)
        else:raise TimeoutError('Continuation queue identity not registered')
        results.append(dict(stage=str(stage), old_owner=old_owner, new_owner=anchor,
            plan_sha256=sha(stage / 'plan.json'), source_sha256=receipt['source_sha256']))
        atomic(ROOT / (task + '-queue-rebound.json'), results[-1])
    assert owned(route['owner']) and sha(CYCLE / 'plan.json') == cycle_sha
    ready = dict(version=1, ok=True, time=time.time(), cycle_id=CYCLE.name, cycle_plan_sha256=cycle_sha,
                 owner=route['owner'], queues=results, stage1_restarted=False, training_config_changed=False)
    atomic(ROOT / 'queue-rebind-ready.json', ready)
    print(json.dumps(ready))


if __name__ == '__main__':
    main()
