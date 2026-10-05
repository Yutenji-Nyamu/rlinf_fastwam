"""One SZ3 GPU4 Rynn diagnostic child, followed by the existing exact RLT return.

prepare writes files only; launch dispatches once. GPU5/6/7 are never borrowed.
This diagnostic return is a single-card provenance record, not a four-card WM
owner that may be blindly passed to the combined returned-cycle preparer.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D = S / 'rynn-diagnosis-v1'
O = D / 'run'
PARENT = S / 'runs/rynn-success-v2'
PLAN = D / 'prepared/plan.json'
SCRIPT = D / 'code/rynn_diagnostic_owner.py'
RYNN_PYTHON = '/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python'
MODEL = Path('/data/chenyiteng/models/RynnValue-8B-8738c5e4')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o600)


def check_host():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'


def validate_contract(plan):
    assert plan['mode'] == 'rynn-single-gpu-diagnostic'
    assert plan['physical_gpus'] == [4] and plan['untouched_gpus'] == [5, 6, 7]
    assert plan['owner_dir'] == str(O) and plan['parent_owner'] == str(PARENT)
    assert plan['timeout_seconds'] == 900
    argv = plan['argv']
    assert argv[0] == RYNN_PYTHON and argv[1:3] == ['-u', '-B']
    assert argv[3] == str(D / 'code/rynn_diagnostic.py')
    expected = {'--service-module': str(S / 'rynn-control-v2/code/rynn_success_service.py'),
        '--model-path': str(MODEL), '--manifest-path': str(MODEL / 'manifest.json'),
        '--physical-gpu': '4', '--cases-json': str(D / 'prepared/cases.json'),
        '--samples-npz': str(D / 'prepared/samples.npz'), '--output': str(O / 'result.json')}
    assert len(argv[4:]) == len(expected) * 2
    for flag, value in expected.items():
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value
    assert plan['environment']['CUDA_VISIBLE_DEVICES'] == '4'
    assert plan['environment']['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert not any(k in plan['environment'] for k in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'))
    assert set(plan['untouched_drivers']) == {'gpu5', 'gpu6', 'gpu7'}


def modules(plan):
    check_host()
    validate_contract(plan)
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, 'Frozen diagnostic input changed: ' + path
    assert plan['source_sha256'][str(SCRIPT)] == sha(__file__)
    R = load('rynn_diagnostic_return', plan['lifecycle_module'])
    stage = Path(plan['lifecycle_path'])
    R.install_helper(stage)
    cp = R.load_plan(stage)
    assert cp['group'] == 'gpu4' and set(cp['runs']) == {'gpu4'}
    assert cp['completed_owner'] == str(PARENT)
    M = load('rynn_diagnostic_process_catalog', plan['catalog_module'])
    M.H = R.H
    M.pidfd_probe()
    return R, M, stage


def prepare():
    check_host()
    assert not PLAN.exists() and not O.exists()
    tested = read(D / 'prepared/cpu-tests.json')
    assert tested['all_cpu_tests_passed'] is True
    for path, digest in tested['source_sha256'].items():
        assert sha(path) == digest
    old = read(PARENT / 'owner-plan.json')
    final = read(PARENT / 'final.json')
    assert final['terminal_status'] in ('completed', 'failed')
    assert final['recovery_error'] is None and final['rlt_return_dispatched'] is True
    assert final['rlt_borrowed'] is True and read(PARENT / 'cleanup.json')['all_stopped'] is True
    helper = Path(old['base_formal_wrapper']).parent / 'rlt_returned_cycle.py'
    catalog = Path(old['base_owner_module'])
    for path in (helper, catalog, S / 'rynn-control-v2/code/rynn_success_service.py'):
        assert old['source_sha256'][str(path)] == sha(path)
    R = load('rynn_diagnostic_prepare_return', helper)
    previous = read(Path(old['lifecycle_path']) / 'plan.json')['children']
    prior = previous['gpu4']
    base = Path(read(Path(prior['path']) / 'plan.json')['base_module'])
    scope = old['graphics_scope']
    active = read(scope['activation_receipt'])
    assert active['status'] == 'active'
    stage = D / 'prepared/rynn-diagnosis-v1-gpu4'
    R.prepare(stage, Path(prior['path']), Path(prior['module']), PARENT, base,
        scope_activation=scope['activation_receipt'], scope_manifest=active['manifest'], scope_id=active['scope_id'])
    untouched = {}
    other = read(Path(previous['gpu567']['path']) / 'plan.json')
    for key, row in other['runs'].items():
        identity = read(Path(row['new_run']) / 'runtime/driver-identity.json')
        assert R.H.same(identity)
        untouched[key] = dict(identity=identity, run=row['new_run'], namespace=row['namespace'],
            lifecycle_path=previous['gpu567']['path'], lifecycle_module=previous['gpu567']['module'])
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL') if k in os.environ}
    env.update(CUDA_VISIBLE_DEVICES='4', CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false')
    argv = [RYNN_PYTHON, '-u', '-B', str(D / 'code/rynn_diagnostic.py'),
        '--service-module', str(S / 'rynn-control-v2/code/rynn_success_service.py'),
        '--model-path', str(MODEL), '--manifest-path', str(MODEL / 'manifest.json'),
        '--physical-gpu', '4', '--cases-json', str(D / 'prepared/cases.json'),
        '--samples-npz', str(D / 'prepared/samples.npz'), '--output', str(O / 'result.json')]
    child = read(stage / 'plan.json')
    files = [SCRIPT, D / 'code/rynn_diagnostic.py', Path(argv[5]), MODEL / 'manifest.json',
        D / 'prepared/cases.json', D / 'prepared/samples.npz', D / 'prepared/cpu-tests.json',
        stage / 'plan.json', catalog, Path(scope['activation_receipt']), Path(active['manifest']), Path(active['runtime_path'])]
    source = dict(child['frozen_files'])
    source.update({str(path): sha(path) for path in files})
    plan = dict(mode='rynn-single-gpu-diagnostic', owner_dir=str(O), parent_owner=str(PARENT),
        lifecycle_path=str(stage), lifecycle_module=str(stage / 'rlt_returned_cycle.py'), catalog_module=str(catalog),
        python=child['python'], source_sha256=source, physical_gpus=[4], untouched_gpus=[5, 6, 7],
        untouched_drivers=untouched, timeout_seconds=900, argv=argv, environment=env)
    validate_contract(plan)
    save(PLAN, plan)
    save(D / 'prepared.json', dict(plan=str(PLAN), plan_sha256=sha(PLAN), preparation_only=True,
        gpu4_checkpoint=child['runs']['gpu4']['recovery']['checkpoint']['step'], untouched_gpus=[5, 6, 7]))
    print(json.dumps(read(D / 'prepared.json')), flush=True)


def launch():
    plan = read(PLAN)
    assert read(D / 'prepared.json')['plan_sha256'] == sha(PLAN)
    R, M, stage = modules(plan)
    assert not O.exists() and not (D / 'launch-attempt.json').exists()
    row = R.load_plan(stage)['runs']['gpu4']
    assert R.H.same(row['original_identity'])
    assert not (stage / 'old-stop-attempt.json').exists()
    assert all(R.H.same(row['identity']) for row in plan['untouched_drivers'].values())
    save(D / 'launch-attempt.json', dict(time=R.H.now(), plan_sha256=sha(PLAN)))
    argv = [plan['python'], '-u', '-B', str(SCRIPT), 'owner']
    with (D / 'owner-launch.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=str(D / 'code'), env=plan['environment'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = R.H.proc(child.pid)
    assert identity and identity['uid'] == 20001
    save(D / 'launch-receipt.json', dict(time=R.H.now(), identity=identity, argv=argv, plan_sha256=sha(PLAN)))
    print(json.dumps(read(D / 'launch-receipt.json')), flush=True)


def check_gpu_scope(H, catalog):
    catalog.scan()
    pids = {row['pid'] for row in catalog.live()}
    contexts = H.gpu_processes(list(range(8)))
    assert not [row for row in contexts if row['pid'] in pids and row['gpu'] != 4], 'Diagnostic escaped GPU4'
    assert not [row for row in contexts if row['gpu'] == 4 and row['pid'] not in pids], 'Foreign GPU4 context'
    return contexts


def cleanup(H, catalog):
    sent = []
    for sig, seconds in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        deadline = time.monotonic() + seconds
        while True:
            catalog.scan()
            live = catalog.live()
            if not live:
                return dict(all_stopped=True, sent=sent, processes=list(catalog.rows.values()))
            for row in live:
                key = (row['pid'], row['start'], int(sig))
                if key not in sent:
                    catalog.send(row, sig)
                    sent.append(key)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    raise RuntimeError('Diagnostic descendants remain; GPU4 return withheld')


def owner():
    import fcntl
    plan = read(PLAN)
    assert read(D / 'prepared.json')['plan_sha256'] == sha(PLAN)
    R, M, stage = modules(plan)
    H = R.H
    O.mkdir(mode=0o700)
    H.save(O / 'owner-plan.json', plan)
    H.save(O / 'owner-identity.json', H.proc(os.getpid()))
    catalog = M.Catalog(O / 'processes.json', str(uuid.uuid4()))
    status, error, recovery_error, exit_code = 'not_started', None, None, None
    borrowed, returned, child = False, False, None
    lock = (stage / 'operation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def terminate(signum, _frame):
        raise RuntimeError('Diagnostic owner received signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        R.stop(stage)
        borrowed = True
        H.save(O / 'borrowed.json', dict(time=H.now(), physical_gpus=[4], cycle_id=stage.name))
        assert not H.gpu_processes([4])
        environment = dict(plan['environment'], **{M.TOKEN: catalog.token, M.PHASE: 'diagnostic'})
        H.save(O / 'child-launch-attempt.json', dict(time=H.now(), argv=plan['argv']))
        with (O / 'diagnostic.log').open('x') as log:
            child = subprocess.Popen(plan['argv'], cwd=str(D / 'code'), env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = H.proc(child.pid)
        assert identity and identity['uid'] == 20001
        # Keep the direct-child receipt even if it has already become a zombie;
        # cleanup/return still needs its exact identity after an immediate error.
        catalog.rows[(identity['pid'], identity['start'])] = dict(identity, phase='diagnostic',
            proof='sole direct child of exact diagnostic owner')
        catalog.write()
        H.save(O / 'child-identity.json', identity)
        status = 'failed'
        deadline = time.monotonic() + plan['timeout_seconds']
        while child.poll() is None:
            contexts = check_gpu_scope(H, catalog)
            H.atomic(O / 'heartbeat.json', dict(time=H.now(), child_alive=True, gpu_contexts=contexts))
            if time.monotonic() >= deadline:
                status = 'timed_out'
                raise TimeoutError('Rynn diagnostic exceeded 900 seconds')
            time.sleep(3)
        exit_code = child.returncode
        assert exit_code == 0, 'Rynn diagnostic child failed: ' + str(exit_code)
        assert (O / 'result.json').is_file(), 'Diagnostic result missing'
        assert read(O / 'result.json')['engineering_passed'] is True
        H.save(O / 'result-receipt.json', dict(path=str(O / 'result.json'), sha256=sha(O / 'result.json')))
        status = 'completed'
    except BaseException as exc:
        error = dict(type=type(exc).__name__, error=str(exc))
        if status != 'timed_out':
            status = 'failed' if child is not None else 'not_started'
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            cleared = cleanup(H, catalog)
            if child is not None:
                child.wait(timeout=5)
                exit_code = child.returncode
            H.save(O / 'cleanup.json', cleared)
            # Complete an already successful scoped stop if its final receipt was interrupted.
            if not (stage / 'rlt-stopped.json').exists() and (stage / 'old-stopped.json').exists():
                R.finalize_stopped(stage)
            borrowed = (stage / 'rlt-stopped.json').exists()
            if borrowed:
                assert not H.gpu_processes([4]), 'GPU4 compute/graphics context remains'
                known = {row['pid'] for row in catalog.rows.values()}
                assert not [row for row in H.gpu_processes(list(range(8))) if row['pid'] in known]
                release = O / 'diagnostic-release.json'
                H.save(release, dict(time=H.now(), cycle_id=stage.name, gpus=[4], terminal_status=status,
                    all_workers_stopped=True, managed_processes=list(catalog.rows.values())))
                result = H.resume(stage, release)
                returned = True
                H.save(O / 'rlt-return-dispatched.json', dict(time=H.now(), result=result, lifecycle_path=str(stage)))
            elif (stage / 'old-stop-attempt.json').exists():
                raise RuntimeError('GPU4 RLT stop incomplete; inspect exact lifecycle before recovery')
        except BaseException as exc:
            recovery_error = dict(type=type(exc).__name__, error=str(exc))
        H.save(O / 'final.json', dict(time=H.now(), terminal_status=status, error=error,
            recovery_error=recovery_error, child_exit_code=exit_code, rlt_borrowed=borrowed,
            rlt_return_dispatched=returned, lifecycle_path=str(stage), physical_gpus=[4],
            first_round_validation_pending=returned, untouched_gpus=[5, 6, 7],
            untouched_drivers_live={key: H.same(row['identity']) for key, row in plan['untouched_drivers'].items()},
            subsequent_borrow_note='GPU4 now comes from this single-card lifecycle; GPU567 retain the V2 parent cycle.'))
        lock.close()
    return 0 if status == 'completed' and recovery_error is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'launch', 'owner'))
    action = parser.parse_args().action
    result = globals()[action]()
    raise SystemExit(result if isinstance(result, int) else 0)
