"""Reuse a verified Stage1, then cut over exactly one existing RLT pair.

Run once as chenyiteng with --stage pointing to a new, reviewed plan. The
unchanged prior ops module handles scoped stopping and formal driver launch.
"""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import socket
import time


ROOT = Path('/data/chenyiteng')
HOSTS = {'admin': ('sz1', 1003), 'h100-gpu01': ('sz3', 20001)}


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def owned(path, uid, directory=False):
    path = Path(path)
    assert path.is_absolute() and not path.is_symlink(), str(path)
    assert (path.is_dir() if directory else path.is_file()), str(path)
    assert path.stat().st_uid == uid and path.resolve().is_relative_to(ROOT.resolve()), str(path)
    return path


def read(path, uid):
    return json.loads(owned(path, uid).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_reused_stage1(plan, uid):
    source = owned(plan['reuse_source_stage'], uid, directory=True)
    assert source.parent == ROOT / 'deployment-20260928'
    old = read(source / 'plan.json', uid)
    assert old['uid'] == uid and old['task'] == plan['task']
    assert plan['runs']['stage1-full'] == old['runs']['stage1-full']
    assert plan['stage1_full_weights'] == old['stage1_full_weights']
    receipt = read(source / 'stage1-complete.json', uid)
    runtime = Path(old['runs']['stage1-full']['run']) / 'runtime'
    finished = read(runtime / 'finished.json', uid)
    assert finished['exit_code'] == 0, 'Original Stage1 did not finish successfully'
    weights = owned(plan['stage1_full_weights'], uid)
    assert weights.parent.name == 'model_state_dict' and weights.parent.parent.name == 'actor'
    assert weights.parents[2].name == 'global_step_2000'
    assert weights.name == 'full_weights.pt' and weights.stat().st_size > 1024 ** 3
    assert receipt['weights'] == str(weights) and receipt['bytes'] == weights.stat().st_size
    return {'time': now(), 'weights': str(weights), 'bytes': weights.stat().st_size,
            'reused_from': str(source), 'source_receipt': str(source / 'stage1-complete.json'),
            'source_finished': str(runtime / 'finished.json'), 'stage1_retrained': False}


def check_old_live(plan, ops, uid):
    assert len(plan['old_runs']) == 2
    assert {row['source_key'] for row in plan['old_runs']} == {'clean', 'combo'}
    for row in plan['old_runs']:
        source = read(row['source_plan'], uid)
        original = source['runs'][row['source_key']]
        assert all(row[key] == original[key] for key in ('run', 'namespace', 'gpus'))
        identity = read(Path(row['run']) / 'runtime/driver-identity.json', uid)
        saved = ops.normalized(row['identity'])
        actual = ops.normalized(identity)
        assert all(actual[key] == saved[key] for key in ('pid', 'uid', 'start'))
        assert actual['uid'] == uid and actual['namespace'] == row['namespace']
        assert ops.same(actual), 'Old driver identity is no longer live; refresh before cutover'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    host, uid = HOSTS[socket.gethostname()]
    assert os.getuid() == uid, 'Run as chenyiteng, not root'
    stage = owned(args.stage, uid, directory=True)
    assert stage.parent == ROOT / 'deployment-20260929'
    plan = read(stage / 'plan.json', uid)
    assert plan['uid'] == uid
    for key in ('clean', 'combo'):
        row = plan['runs'][key]
        assert row['namespace'] == f"fx8full-{host}-{plan['task']}-{key}-0929"
        runtime = owned(Path(row['run']) / 'runtime', uid, directory=True)
        assert not any((runtime / name).exists() for name in
                       ('driver.log', 'launch.json', 'driver-identity.json', 'finished.json'))
    assert not any((stage / name).exists() for name in
                   ('formal-only-attempt.json', 'pipeline-identity.json', 'old-stop-attempt.json',
                    'old-stopped.json', 'formal-dispatched.json', 'pipeline-finished.json'))
    spec = importlib.util.spec_from_file_location('n8full_prior_ops', owned(plan['ops'], uid))
    assert spec and spec.loader
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    ops.ST = stage
    assert ops.checked() == plan
    reuse = check_reused_stage1(plan, uid)
    check_old_live(plan, ops, uid)
    # Claim this exact cutover before any signal or child process is issued.
    save(stage / 'formal-only-attempt.json', {'time': now(), 'host': host,
         'plan': str(stage / 'plan.json'), 'source_stage': plan['reuse_source_stage'],
         'old_runs': [row['run'] for row in plan['old_runs']], 'stage1_retrained': False})
    save(stage / 'pipeline-identity.json', {**ops.proc(os.getpid()), 'time': now()})
    try:
        save(stage / 'stage1-complete.json', reuse)
        check_old_live(plan, ops, uid)
        try:
            ops.stop_old()
        except AssertionError as error:
            if str(error) != 'Old GPU processes remain':
                raise
            # A stopped CUDA context can take longer to disappear from nvidia-smi.
            # Wait only; never replay the old stop or send an additional signal.
            assert (stage / 'old-stop-attempt.json').is_file()
            assert not (stage / 'old-stopped.json').exists()
            gpus = {gpu for row in plan['old_runs'] for gpu in row['gpus']}
            for _ in range(90):
                assert not any(ops.same(row['identity']) for row in plan['old_runs'])
                assert not any(ops.active_actors(plan, row['namespace']) for row in plan['old_runs'])
                if not ops.gpu_pids(gpus):
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Old CUDA contexts did not release; no automatic restart')
            save(stage / 'old-stopped.json', {'time': now(),
                 'runs': [row['run'] for row in plan['old_runs']],
                 'context_release_wait': True, 'additional_signals': False})
        children = {}
        for key in ('clean', 'combo'):
            children[key] = ops.launch(key)
            save(stage / ('formal-launched-' + key + '.json'),
                 {'time': now(), 'identity': ops.proc(children[key].pid), 'role': key})
        save(stage / 'formal-dispatched.json', {'time': now(),
             'drivers': {key: ops.proc(child.pid) for key, child in children.items()},
             'stage1_reused': True})
        failures = {}
        while children:
            for key, child in list(children.items()):
                exit_code = child.poll()
                if exit_code is not None:
                    ops.finished(key, exit_code)
                    if exit_code:
                        failures[key] = exit_code
                    del children[key]
            if children:
                time.sleep(10)
        save(stage / 'pipeline-finished.json', {'time': now(), 'failures': failures,
                                               'stage1_reused': True})
        assert not failures, failures
    except BaseException as error:
        save(stage / 'formal-only-error.json', {'time': now(),
             'error_type': type(error).__name__, 'error': str(error)[:2000],
             'started_drivers': {key: ops.proc(child.pid) for key, child in locals().get('children', {}).items()},
             'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
