"""Bounded native diagnosis -> fixed lifecycle check -> resumed formal EXPO.

One owner borrows SZ2 GPUs 4--7 and restores the exact current RLT lineage only
after every managed child and GPU allocation is gone. Shared Ray is untouched.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time
import traceback
import uuid

from expo_smoke_owner import Roster, atomic, gpu_map, gpu_processes, identity, owned
from expo_process import pidfd_probe

ROOT = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002')
SOURCE = ROOT / 'source'
CYCLE = ROOT / 'rlt-cycle-expo-turn-switch-repair-20261002-v1'
RLT_PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
EXPO_PY = '/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python'
UUIDS = ['GPU-a0a252d6-828d-29e1-1fd2-65187f573f4d', 'GPU-2cd891ea-180d-da39-6419-2d7033f8b21b',
         'GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4', 'GPU-3c6321c1-3e58-c071-3867-533391152fe7']
OWNER_FILES = ('expo_formal_owner.py', 'expo_formal_resources.py', 'expo_rlt_switch.py',
               'expo_smoke_owner.py', 'expo_process.py')


class Terminated(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe(inputs):
    result = copy.deepcopy(inputs)
    result.pop('port_source_manifest', None)
    return result


def verify_inputs(expected_recipe=None):
    inputs = read(ROOT / 'inputs.json')
    assert inputs['formal']['max_physical_actions'] == 20000
    if expected_recipe is not None:
        assert recipe(inputs) == expected_recipe, 'Training recipe changed during diagnostic gate'
    manifest = inputs['port_source_manifest']
    assert all('tools/' + name in manifest for name in OWNER_FILES)
    for name, expected in manifest.items():
        path = (SOURCE / name).resolve(strict=True)
        assert path.is_relative_to(SOURCE.resolve()) and sha(path) == expected, name
    return inputs


def native_receipt():
    receipt = read(ROOT / 'native-check.json')
    assert receipt['ok'] is True and receipt['mode'] == 'fixed'
    assert receipt['inputs_sha256'] == sha(ROOT / 'inputs.json')
    assert receipt['test_sha256'] == sha(SOURCE / 'tests/native_expo_lifecycle.py')
    return receipt


def diagnostic_command(path):
    assert path.resolve(strict=True) == (ROOT / 'diagnostic.json').resolve(strict=True)
    config = read(path)
    command = config['argv']
    assert isinstance(command, list) and all(isinstance(word, str) for word in command)
    assert command[:5] == [EXPO_PY, '-X', 'faulthandler', '-u', '-B']
    script = Path(command[5]).resolve(strict=True)
    assert script.is_relative_to((SOURCE / 'tests').resolve(strict=True)) and script.suffix == '.py'
    manifest = config['source_sha256']
    assert str(script.relative_to(SOURCE.resolve(strict=True))) in manifest
    for name, expected in manifest.items():
        target = (SOURCE / name).resolve(strict=True)
        assert target.is_relative_to(SOURCE.resolve()) and sha(target) == expected
    timeout = config.get('timeout_seconds', 900)
    assert isinstance(timeout, (int, float)) and 30 <= timeout <= 1800
    return command, timeout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('formal', 'diagnostic'), default='formal')
    parser.add_argument('--diagnostic-config', type=Path, default=ROOT / 'diagnostic.json')
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    pidfd_probe()
    # Scope crash handling to this process tree; do not alter the host kernel.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    temporary = ROOT / 'tmp'
    temporary.mkdir(mode=0o700, exist_ok=True)
    with (ROOT / 'resource-owner.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (ROOT / 'owner.json').exists(), 'Owner was already launched; inspect receipts'
        scope = 'expo-repair-' + str(uuid.uuid4())
        me = identity(os.getpid())
        metadata = dict(owner=me, scope=scope, physical_gpus=[4, 5, 6, 7], status='PREPARING', time=time.time())
        atomic(ROOT / 'owner.json', metadata)
        atomic(ROOT / 'current.json', metadata)
        requested = [False]
        closing = [False]
        critical = [False]

        def terminate(sig, frame):
            requested[0] = True
            if not closing[0] and not critical[0]:
                raise Terminated('Owner signal ' + str(sig))

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)

        def state(name, **more):
            metadata.update(status=name, time=time.time(), **more)
            atomic(ROOT / 'current.json', metadata)

        def operation(action, release=None):
            intent = ROOT / (action + '-intent.json')
            assert not intent.exists(), 'Refusing repeated ' + action
            atomic(intent, dict(action=action, time=time.time(), owner=me))
            command = [RLT_PY, '-X', 'faulthandler', '-u', '-B', str(SOURCE / 'tools/expo_formal_resources.py'), action]
            if release:
                command += ['--release', str(release)]
            critical[0] = True
            try:
                with (ROOT / (action + '.log')).open('xb') as log:
                    subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            finally:
                critical[0] = False

        roster = None
        child = None
        managed = []
        error = None
        stopped = released = passed = False
        return_status = None
        owner_hashes = {name: sha(SOURCE / 'tools' / name) for name in OWNER_FILES}

        def fixed_owner():
            assert owner_hashes == {name: sha(SOURCE / 'tools' / name) for name in OWNER_FILES}, 'Live owner/helper source changed'

        def cleanup_child(graceful=False):
            nonlocal child, roster
            if roster is None:
                return
            if graceful and child.poll() is None and owned(roster.root):
                roster.signal(roster.root, signal.SIGTERM)
                deadline = time.monotonic() + 180
                while child.poll() is None and time.monotonic() < deadline:
                    roster.scan()
                    time.sleep(1)
            result = roster.cleanup()
            atomic(ROOT / ('cleanup-' + roster.phase + '.json'), result)
            managed.extend(roster.rows.values())
            child.wait(timeout=10)
            assert not any(row['gpu'] in UUIDS for row in gpu_processes()), 'Managed phase did not release GPUs 4--7'
            child = roster = None

        def run_child(command, phase, timeout=None, heartbeat=False):
            nonlocal child, roster
            assert roster is None and child is None
            fixed_owner()
            assert not any(row['gpu'] in UUIDS for row in gpu_processes())
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(UUIDS),
                PYTHONPATH=str(SOURCE) + ':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
                LD_LIBRARY_PATH='/home/chenyiteng/tools/cuda-12.9/lib64',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1', MPLBACKEND='Agg',
                TMPDIR=str(temporary),
                EXPO_SMOKE_OWNER_SCOPE=scope, EXPO_SMOKE_OWNER_PHASE=phase, PYTHONDONTWRITEBYTECODE='1')
            (ROOT / 'driver-heartbeat').touch()
            critical[0] = True
            try:
                with (ROOT / (phase + '.log')).open('xb') as log:
                    child = subprocess.Popen(command, cwd=SOURCE, env=environment, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    anchor = identity(child.pid)
                    roster = Roster(anchor, scope, phase, ROOT / ('process-roster-' + phase + '.json'))
                    roster.write()
            finally:
                critical[0] = False
            state(phase.upper() + '_RUNNING', child=anchor, command=command)
            started = last_sample = time.monotonic()
            while child.poll() is None:
                if requested[0]:
                    raise Terminated('Requested while ' + phase)
                roster.scan()
                if timeout is not None and time.monotonic() - started > timeout:
                    raise TimeoutError(phase + ' exceeded ' + str(timeout) + ' seconds')
                if heartbeat and time.time() - (ROOT / 'driver-heartbeat').stat().st_mtime > 900:
                    raise TimeoutError('EXPO driver heartbeat stale >15min')
                if time.monotonic() - last_sample > 30:
                    sample = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                        '--format=csv,noheader'], text=True, timeout=20)
                    with (ROOT / 'gpu-samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(time=time.time(), phase=phase, gpu=sample)) + '\n')
                    last_sample = time.monotonic()
                time.sleep(2)
            result = child.returncode
            cleanup_child()
            atomic(ROOT / (phase + '-exit.json'), dict(returncode=result, time=time.time(), gpu_released=True))
            return result

        try:
            assert [gpu_map().get(i) for i in (4, 5, 6, 7)] == UUIDS
            inputs = verify_inputs()
            expected_recipe = recipe(inputs)
            if args.mode == 'diagnostic':
                diagnosis, diagnosis_timeout = diagnostic_command(args.diagnostic_config)
            operation('prepare')
            if requested[0]:
                raise Terminated('Requested during preparation')
            state('STOPPING_RLT')
            operation('stop')
            proof = read(CYCLE / 'rlt-stopped.json')
            assert proof['all_original_drivers_stopped'] and proof['all_original_namespaces_empty']
            assert set(proof['gpus_released']) == {4, 5, 6, 7}
            stopped = True
            if requested[0]:
                raise Terminated('Requested during RLT stop')
            if args.mode == 'diagnostic':
                result = run_child(diagnosis, 'diagnostic', diagnosis_timeout)
                # A reproducer may intentionally segfault; gate requires an
                # explicit fresh repair receipt before any formal launch.
                state('WAITING_FOR_REPAIR_READY', diagnostic_returncode=result)
                deadline = time.monotonic() + 1800
                while not (ROOT / 'ready.json').exists():
                    if requested[0]:
                        raise Terminated('Requested at repair gate')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Repair ready gate exceeded 30min')
                    heartbeat_file = ROOT / 'controller-heartbeat'
                    if not heartbeat_file.exists() or time.time() - heartbeat_file.stat().st_mtime > 300:
                        raise TimeoutError('Repair controller heartbeat stale >5min')
                    fixed_owner()
                    time.sleep(2)
                ready = read(ROOT / 'ready.json')
                assert ready['ok'] is True and ready['inputs_sha256'] == sha(ROOT / 'inputs.json')
            fixed_owner()
            verify_inputs(expected_recipe)
            assert (ROOT / 'run/checkpoint-latest.pt').is_file(), 'Resumed formal checkpoint must be migrated before ready'
            test = SOURCE / 'tests/native_expo_lifecycle.py'
            command = [EXPO_PY, '-X', 'faulthandler', '-u', '-B', str(test), '--mode', 'fixed']
            assert run_child(command, 'native-fixed', 1200) == 0, 'Fixed native lifecycle failed'
            receipt = native_receipt()
            state('NATIVE_FIXED_VERIFIED', native_receipt=receipt)
            verify_inputs(expected_recipe)
            command = [EXPO_PY, '-X', 'faulthandler', '-u', '-B',
                str(SOURCE / 'examples/embodiment/train_expo_formal.py'), '--inputs', str(ROOT / 'inputs.json'),
                '--run', str(ROOT / 'run'), '--max-physical-actions', '20000', '--enable-evaluation',
                '--resume', str(ROOT / 'run/checkpoint-latest.pt')]
            result = run_child(command, 'driver', heartbeat=True)
            if result:
                raise RuntimeError('EXPO formal exited ' + str(result) + '; inspect driver.log')
            complete = read(ROOT / 'run/complete.json')
            assert complete['ok'] and complete['budget_completed']
            assert complete['cadence']['counters']['physical_actions'] == 20000
            assert complete['cadence']['counters']['pending_calls'] == 0
            passed = True
        except BaseException as failure:
            error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
        finally:
            closing[0] = True
            try:
                if roster:
                    state('CLEANING_EXPO')
                    cleanup_child(graceful=roster.phase == 'driver')
                if not stopped and (CYCLE / 'rlt-stopped.json').exists():
                    proof = read(CYCLE / 'rlt-stopped.json')
                    stopped = bool(proof.get('all_original_drivers_stopped') and proof.get('all_original_namespaces_empty')
                        and set(proof.get('gpus_released', [])) == {4, 5, 6, 7})
                if stopped:
                    released = not any(row['gpu'] in UUIDS for row in gpu_processes())
                    assert released, 'EXPO GPU remains; refuse RLT resume'
                    fixed_owner()
                    release = dict(cycle_id=CYCLE.name, terminal_status='completed' if passed else ('failed' if managed else 'not_started'),
                        all_workers_stopped=True, managed_processes=managed, physical_gpus=[4, 5, 6, 7])
                    atomic(ROOT / 'release.json', release)
                    state('RESTORING_RLT')
                    operation('resume', ROOT / 'release.json')
                    deadline = time.monotonic() + 1800
                    while time.monotonic() < deadline:
                        with (ROOT / 'rlt-status.log').open('w') as log:
                            subprocess.run([RLT_PY, '-X', 'faulthandler', '-u', '-B',
                                str(SOURCE / 'tools/expo_formal_resources.py'), 'status'], cwd=SOURCE,
                                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
                        status = json.loads((ROOT / 'rlt-status.log').read_text().splitlines()[-1])
                        atomic(ROOT / 'rlt-status.json', status)
                        if status.get('all_first_rounds_verified'):
                            return_status = status
                            break
                        time.sleep(20)
                    assert return_status, 'RLT dispatched; first-round verification still needs action'
                    state('RLT_RESTORED', formal_passed=passed)
                elif (ROOT / 'stop-intent.json').exists():
                    raise RuntimeError('Partial RLT stop without full receipt; inspect stop.log before recovery')
            except BaseException as failure:
                error = dict(type='CLEANUP_OR_RLT_RETURN_NEEDS_ACTION', message=str(failure), previous=error,
                    traceback=traceback.format_exc())
                state('ACTION_REQUIRED')
            atomic(ROOT / 'final.json', dict(time=time.time(), formal_passed=passed, error=error,
                gpu_released=released, rlt_first_rounds_verified=bool(return_status), owner=me, scope=scope))
        return 0 if passed and return_status and error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
