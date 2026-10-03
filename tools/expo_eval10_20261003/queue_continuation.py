"""Resume two already-trained RLT queues after an explicitly rebound EXPO gate.

This new wrapper never launches or waits for Stage1. It keeps the original
queue owner's gate, scoped stop, launch, watch update and monitoring behavior.
The original owner identity and all Stage1 receipts remain unchanged.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


CTRL = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/eval10-continuation-20261003')
CYCLE_ID = 'rlt-cycle-expo-eval10-20261003-v1'
OPS_SHA256 = '1ef808a8296c2672c1c6841432db19f9a25a99f6b434259018184059de633ff7'
QUEUE_SHA256 = '4d2d2f8733f3e2db6c5c040a4ab10d8578bc946b0e9ee673692414c07abcbed7'
TASK_GPUS = {'place_object_stand': (4, 5), 'move_playingcard_away': (6, 7)}
MINIMUM_STAGE1_BYTES = 1024 ** 3


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_equal(left, right):
    return all(left.get(k) == right.get(k) for k in ('pid', 'uid', 'start'))


def validate_plan_transition(previous, plan, cycle, cycle_sha, new_owner):
    """Pure checks: no model/config/run/budget edit is permitted in a rebind."""
    assert set(previous) == set(plan), 'Plan fields changed'
    assert {k: v for k, v in previous.items() if k not in ('gates', 'old_runs')} == {
        k: v for k, v in plan.items() if k not in ('gates', 'old_runs')
    }, 'Only gates and old_runs may change'
    assert plan['task'] in TASK_GPUS and plan['uid'] == 20001
    assert set(plan['gates']) == {'clean', 'combo'}
    assert cycle['cycle_id'] == CYCLE_ID, 'Wrong continuation cycle'
    assert len(plan['old_runs']) == 2
    for role, gpu in zip(('clean', 'combo'), TASK_GPUS[plan['task']]):
        assert plan['runs'][role]['gpus'] == [gpu], 'Formal GPU changed'
        spec = plan['gates'][role]
        assert spec['kind'] == 'global' and spec['gpu'] == gpu
        assert Path(spec['cycle_plan']).resolve() == (CTRL / CYCLE_ID / 'plan.json').resolve()
        assert spec['cycle_sha256'] == cycle_sha, 'Cycle hash changed'
        assert Path(spec['final']).resolve() == (CTRL / 'final.json').resolve()
        assert spec['receipts'] == [str(CTRL / 'rlt-status.json')]
        assert identity_equal(spec['owner'], new_owner), 'Borrow owner differs'
        assert new_owner['uid'] == plan['uid']
        targets = [x for x in plan['old_runs'] if x['gpus'] == [gpu]]
        assert len(targets) == 1, 'Ambiguous return target'
        target = targets[0]
        expected = cycle['runs']['gpu' + str(gpu)]
        assert target['run'] == expected['new_run'], 'Wrong returned run'
        assert target['namespace'] == expected['namespace'], 'Wrong returned namespace'
        assert target.get('identity') is None, 'Returned identity must be read from final proof'


def validate_completed_stage(stage, plan, old_owner, ops):
    """Read-only refusal gates before claiming ownership or allowing cutover."""
    stage = Path(stage)
    recorded = read(stage / 'owner-identity.json')
    assert recorded == old_owner, 'Original queue owner receipt changed'
    assert recorded['boot_id'] == plan['boot_id']
    assert not ops.same(recorded), 'Original queue owner is still alive'
    state = read(stage / 'queue-status.json')
    assert state['task'] == plan['task'] and state['stage1'] == 'COMPLETE'
    assert state['roles'] == {'clean': 'WAITING_BORROW_RETURN', 'combo': 'WAITING_BORROW_RETURN'}
    assert 'error' not in state and 'failures' not in state
    assert not (stage / 'owner-identity-continuation.json').exists(), 'Never replay a continuation'
    for name in ('queue-failure.json', 'queue-finished.json', 'pipeline-identity.json',
                 'formal-dispatched.json', 'pipeline-finished.json'):
        assert not (stage / name).exists(), 'Queue already has terminal or pipeline side effects: ' + name
    assert not [p.name for p in stage.iterdir() if p.name.startswith(('clean-', 'combo-'))], \
        'Formal role already has side effects'
    for role in ('clean', 'combo'):
        run = Path(plan['runs'][role]['run'])
        for name in ('driver.log', 'driver-identity.json', 'launch.json', 'finished.json',
                     'exit_code.txt', 'cleanup-targets.json'):
            assert not (run / 'runtime' / name).exists(), 'Formal driver already touched: ' + role + '/' + name
        for base in (run, run / run.name):
            for name in ('tensorboard', 'checkpoints'):
                assert not (base / name).exists(), 'Formal training output already exists'
    complete = read(stage / 'stage1-complete.json')
    weights = Path(plan['stage1_full_weights'])
    assert Path(complete['weights']).resolve() == weights.resolve(), 'Stage1 weights path changed'
    assert weights.is_file() and not weights.is_symlink(), 'Stage1 full weights missing'
    stat = weights.stat()
    assert stat.st_size == complete['bytes'] and stat.st_size > MINIMUM_STAGE1_BYTES, \
        'Stage1 full weights size changed/incomplete'
    assert stat.st_uid == plan['uid'], 'Foreign Stage1 weights owner'
    runtime = Path(plan['runs']['stage1-full']['run']) / 'runtime'
    assert read(runtime / 'finished.json')['exit_code'] == 0, 'Stage1 did not exit successfully'
    assert (runtime / 'exit_code.txt').read_text().strip() == '0'
    assert not ops.same(read(runtime / 'driver-identity.json')), 'Stage1 driver is still alive'
    return {'stage1_complete_sha256': sha(stage / 'stage1-complete.json'),
            'weights': str(weights), 'bytes': stat.st_size,
            'weights_identity': {'dev': stat.st_dev, 'ino': stat.st_ino, 'uid': stat.st_uid,
                                 'mtime_ns': stat.st_mtime_ns, 'size': stat.st_size}}


def load_original(plan, stage):
    assert sha(plan['ops']) == OPS_SHA256, 'Original ops source changed'
    assert sha(plan['queue_owner']) == QUEUE_SHA256, 'Original queue source changed'
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    ops = load('ops', plan['ops'])
    ops.ST = Path(stage)
    original = load('_expo_previous_queue', plan['queue_owner'])
    assert original.ops is ops
    return ops, original


def role_iteration(plan, state, children, first, failures, ops, original, sleep=time.sleep):
    """The original post-Stage1 role loop; there is no Stage1 launch branch."""
    for role in ('clean', 'combo'):
        if role in children or role in failures or state['roles'][role] == 'COMPLETE':
            continue
        try:
            target = original.gate(plan, role)
            if target is None:
                state['roles'][role] = 'WAITING_BORROW_RETURN'
                continue
            ops.save(ops.ST / f'{role}-cutover-bound.json', {'time': ops.now(), 'target': target})
            ops.stop_old([target], role + '-')
            child = ops.launch(role)
            children[role] = child
            state['roles'][role] = 'STARTING'
            ops.save(ops.ST / f'{role}-dispatched.json', {'time': ops.now(), 'identity': ops.proc(child.pid)})
            for _ in range(60):
                if (Path(plan['runs'][role]['run']) / 'runtime/driver-identity.json').is_file():
                    break
                if child.poll() is not None:
                    raise RuntimeError(f'{role} driver exited on startup')
                sleep(1)
            else:
                raise RuntimeError(f'{role} driver identity missing')
            original.update_watch(plan, role)
        except Exception:
            failures[role] = traceback.format_exc()
            state['roles'][role] = 'NEEDS_ATTENTION'
            ops.save(ops.ST / f'{role}-failure.json', {'time': ops.now(), 'error': failures[role]})
    for role, child in list(children.items()):
        code = child.poll()
        if code is not None:
            ops.finished(role, code)
            state['roles'][role] = 'COMPLETE' if code == 0 else 'FAILED'
            if code:
                failures[role] = f'exit {code}'
            del children[role]
        elif role not in first:
            result = original.first_collection(plan['runs'][role])
            if result:
                ops.save(ops.ST / f'{role}-first-round.json', {'time': ops.now(), 'metrics': result})
                first.add(role)
                state['roles'][role] = 'TRAINING'


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--previous-plan', required=True)
    parser.add_argument('--old-owner-receipt', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    stage = Path(args.stage).resolve()
    plan = read(stage / 'plan.json')
    previous = read(args.previous_plan)
    old_owner = read(args.old_owner_receipt)
    assert plan['task'] in TASK_GPUS
    expected_stage = Path('/data/chenyiteng/deployment-20261002') / ('rlt-next6-' + plan['task'])
    assert stage == expected_stage.resolve(), 'Wrong stage path'
    assert Path(args.previous_plan).resolve() == (CTRL / 'queue-backups' / (plan['task'] + '-plan.json')).resolve()
    assert Path(args.old_owner_receipt).resolve() == (CTRL / 'queue-backups' / (plan['task'] + '-owner-identity.json')).resolve()
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['boot_id']
    ops, original = load_original(plan, stage)
    # The old owner held this same lock. Do not rely only on a process snapshot.
    with (stage / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert ops.checked() == plan
        cycle_path = CTRL / CYCLE_ID / 'plan.json'
        cycle = read(cycle_path)
        new_owner = plan['gates']['clean']['owner']
        validate_plan_transition(previous, plan, cycle, sha(cycle_path), new_owner)
        assert ops.same(new_owner), 'New borrowing owner must be alive for registration'
        proof = validate_completed_stage(stage, plan, old_owner, ops)
        plan_sha = sha(stage / 'plan.json')
        previous_sha = sha(args.previous_plan)
        identity = {**ops.proc(os.getpid()), 'time': ops.now(), 'boot_id': plan['boot_id'],
                    'task': plan['task'], 'stage': str(stage), 'old_owner': old_owner,
                    'plan_sha256': plan_sha, 'previous_plan_sha256': previous_sha,
                    'source_sha256': {str(Path(__file__).resolve()): sha(__file__),
                                      plan['ops']: OPS_SHA256, plan['queue_owner']: QUEUE_SHA256},
                    'stage1_reused': proof, 'stage1_relaunched': False}
        if args.check_only:
            print(json.dumps({'ok': True, 'check_only': True, 'proof': identity}))
            return
        ops.save(stage / 'owner-identity-continuation.json', identity)
        state = {'time': ops.now(), 'task': plan['task'], 'stage1': 'COMPLETE',
                 'roles': {k: 'WAITING_BORROW_RETURN' for k in ('clean', 'combo')},
                 'continuation_owner': {k: identity[k] for k in ('pid', 'uid', 'start')},
                 'stage1_reused': True}
        children, first, failures = {}, set(), {}
        def publish():
            state['time'] = ops.now()
            original.atomic(stage / 'queue-status.json', state)
        publish()
        try:
            while True:
                assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['boot_id']
                assert sha(stage / 'plan.json') == plan_sha, 'Bound queue plan changed'
                role_iteration(plan, state, children, first, failures, ops, original)
                publish()
                if all(state['roles'][r] in ('COMPLETE', 'FAILED', 'NEEDS_ATTENTION') for r in ('clean', 'combo')):
                    break
                time.sleep(30)
            ops.save(stage / 'queue-finished.json', {'time': ops.now(), 'failures': failures})
        except BaseException:
            state['error'] = traceback.format_exc()
            publish()
            ops.save(stage / 'queue-failure.json', {'time': ops.now(), 'error': state['error']})
            raise


if __name__ == '__main__':
    main()
