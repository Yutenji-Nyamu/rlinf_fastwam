"""One-shot same-cycle EXPO graphics repair; retain queue gates and RLT return.

CPU owner only. Contract preparation and the graphics probe precede launch.
Unexpected maintenance errors hold the existing resource lock for inspection;
an identity-pinned abort-return request explicitly executes the original return.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
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
import traceback
import xml.etree.ElementTree as ET

TRAIN = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002')
CTRL = TRAIN.parent / 'eval10-continuation-20261003'
SOURCE = TRAIN / 'source'
CYCLE = CTRL / 'rlt-cycle-expo-eval10-20261003-v1'
EXPO_PY = '/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python'
RLT_PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
UUIDS = ['GPU-a0a252d6-828d-29e1-1fd2-65187f573f4d', 'GPU-2cd891ea-180d-da39-6419-2d7033f8b21b',
         'GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4', 'GPU-3c6321c1-3e58-c071-3867-533391152fe7']
QUEUES = [Path('/data/chenyiteng/deployment-20261002/rlt-next6-' + name)
          for name in ('place_object_stand', 'move_playingcard_away')]
PHASE = 'driver-gpu4567'
KEYS = ('pid', 'uid', 'boot_id', 'start_ticks')
TERMINAL = ('final.json', 'release.json', 'resume-intent.json')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(left, right):
    return all(left.get(key) == right.get(key) for key in KEYS)


def short_same(left, right):
    return (left.get('pid'), left.get('uid'), left.get('start_ticks', left.get('start'))) == (
        right.get('pid'), right.get('uid'), right.get('start_ticks', right.get('start')))


def driver_command():
    return [EXPO_PY, '-X', 'faulthandler', '-u', '-B', str(SOURCE / 'examples/embodiment/train_expo_formal.py'),
            '--inputs', str(TRAIN / 'inputs.json'), '--run', str(TRAIN / 'run'),
            '--max-physical-actions', '20000', '--enable-evaluation',
            '--resume', str(TRAIN / 'run/checkpoint-latest.pt')]


def child_environment(stage, scope):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(UUIDS),
                PYTHONPATH=str(stage / 'bootstrap') + ':' + str(SOURCE) +
                ':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
                LD_LIBRARY_PATH='/home/chenyiteng/tools/cuda-12.9/lib64',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1', PYTHONDONTWRITEBYTECODE='1',
                MPLBACKEND='Agg', TMPDIR=str(CTRL / 'tmp'),
                EXPO_SMOKE_OWNER_SCOPE=scope, EXPO_SMOKE_OWNER_PHASE=PHASE,
                RLINF_EXPO_GPU_SCOPE_MANIFEST=str(stage / 'scope.json'), __GL_APPLICATION_PROFILE='1')
    environment.pop('DISPLAY', None)
    return environment


def validate_contract(contract, stage):
    require(contract.get('version') == 1 and contract.get('purpose') == 'expo-gpu4567-same-cycle', 'Wrong repair contract')
    require(contract.get('stage') == str(stage) and contract.get('control') == str(CTRL)
            and contract.get('train') == str(TRAIN) and contract.get('cycle') == str(CYCLE), 'Wrong repair paths')
    old = contract['old_current']
    require(old.get('status') == 'DRIVER_RUNNING' and old.get('cycle_id') == CYCLE.name
            and old.get('physical_gpus') == [4, 5, 6, 7] and old.get('command') == driver_command(), 'Changed live run contract')
    for row in (old['owner'], old['child']):
        require(row.get('uid') == 20001 and type(row.get('pid')) is int and row['pid'] > 1
                and type(row.get('start_ticks')) is int and row['start_ticks'] > 0
                and bool(row.get('boot_id')), 'Missing exact process identity')
    require(old['owner']['boot_id'] == old['child']['boot_id'], 'Owner/child boot differs')
    require(set(contract['queue_plan_sha256']) == {str(path / 'plan.json') for path in QUEUES}, 'Wrong waiting queues')
    require(len(contract['inputs_sha256']) == 64 and len(contract['cycle_plan_sha256']) == 64, 'Missing input/cycle pin')
    pins = contract['files_sha256']
    resolved_pins = {Path(name).resolve() for name in pins}
    for path in (Path(__file__).resolve(), stage / 'bootstrap/sitecustomize.py', stage / 'scope.json',
                 CTRL / 'tools/reborrow_resources.py', SOURCE / 'tools/expo_smoke_owner.py',
                 SOURCE / 'tools/expo_process.py', SOURCE / 'tools/expo_formal_resources.py',
                 SOURCE / 'tools/expo_rlt_switch.py'):
        require(path.resolve() in resolved_pins, 'Required source pin absent: ' + str(path))
    require(contract.get('graphics_probe_verified') is True, 'Graphics probe was not accepted')


def pins_match(pins):
    for name, digest in pins.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.stat().st_uid == 20001 and any(path.resolve().is_relative_to(root) for root in
                    (Path('/data/chenyiteng').resolve(strict=True), Path('/home/chenyiteng').resolve(strict=True))),
                'Source pin escaped owned data tree: ' + name)
        require(sha(path) == digest, 'Pinned source differs: ' + name)


def all_gpu_processes():
    xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=30)
    result = []
    for index, gpu in enumerate(ET.fromstring(xml).findall('gpu')):
        for process in gpu.findall('./processes/process_info'):
            result.append(dict(index=index, gpu=gpu.findtext('uuid'), pid=int(process.findtext('pid')),
                               type=process.findtext('type'), memory=process.findtext('used_memory')))
    return result


def wait_release(managed, owned, timeout=120):
    deadline = time.monotonic() + timeout
    empty_since = None
    pids = {row['pid'] for row in managed}
    while time.monotonic() < deadline:
        rows = all_gpu_processes()
        # A stale PID after exit can linger in NVML; a reused live PID must never be signalled.
        blocked = [row for row in rows if row['gpu'] in UUIDS or row['pid'] in pids]
        if not blocked and not any(owned(row) for row in managed):
            if empty_since is None:
                empty_since = time.monotonic()
            elif time.monotonic() - empty_since >= 2:
                return
        else:
            empty_since = None
        time.sleep(1)
    raise TimeoutError('Exact EXPO compute/graphics contexts did not settle after cleanup')


def request_valid(value, contract_sha, owner):
    return (value.get('contract_sha256') == contract_sha and value.get('action') in ('resume', 'abort-return')
            and same(value.get('owner', {}), owner))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    # Keep the logical /data route in receipts; validate its resolved location separately.
    stage = Path(os.path.abspath(args.stage))
    require(os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02', 'Wrong host/UID')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Maintenance owner must remain CPU-only')
    require(stage.resolve(strict=True) == (TRAIN.parent / 'gpu4567-fix-20261003').resolve(strict=True)
            and not args.stage.is_symlink()
            and stage.stat().st_uid == 20001, 'Stage must be the exact approved owned repair directory')
    contract_path = stage / 'snapshot-contract.json'
    contract = read(contract_path)
    validate_contract(contract, stage)
    contract_sha = sha(contract_path)
    sys.path.insert(0, str(SOURCE / 'tools'))
    from expo_smoke_owner import Roster, atomic, identity, owned, gpu_map
    from expo_process import pidfd_open, pidfd_send, pidfd_probe
    pidfd_probe()
    me = identity(os.getpid())
    old = contract['old_current']
    old_scope = old['scope']
    new_scope = old_scope + '-gpu4567'
    metadata = dict(owner=me, previous_owner=old['owner'], scope=old_scope, child=old['child'],
                    physical_gpus=[4, 5, 6, 7], cycle_id=CYCLE.name, train_root=str(TRAIN),
                    evaluation_every_episodes=10, evaluation_episodes=20, maintenance=str(stage),
                    contract_sha256=contract_sha, command=driver_command())
    held = retired = False
    active = None
    rosters = []
    managed = []
    popped = None
    requested = [False]
    return_started = False

    def terminate(_sig, _frame):
        requested[0] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminate)

    def signal_exact(row, sig):
        require(owned(row), 'Signal target is no longer exact')
        fd = pidfd_open(row['pid'])
        try:
            require(owned(row), 'Signal target changed before pidfd send')
            pidfd_send(fd, sig)
        finally:
            os.close(fd)

    def wait_stopped(row):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            require(owned(row), 'Held process exited')
            if identity(row['pid'])['state'] in ('T', 't'):
                return
            time.sleep(.05)
        raise TimeoutError('Exact process did not enter stopped state')

    def state(name, **more):
        metadata.update(status=name, time=time.time(), **more)
        atomic(stage / 'status.json', metadata)
        if retired:
            atomic(CTRL / 'current.json', metadata)

    def fixed():
        require(sha(contract_path) == contract_sha, 'Snapshot contract changed')
        pins_match(contract['files_sha256'])
        require(sha(TRAIN / 'inputs.json') == contract['inputs_sha256'], 'Training inputs changed')
        require(sha(CYCLE / 'plan.json') == contract['cycle_plan_sha256'], 'Borrow cycle changed')
        for name, digest in contract['queue_plan_sha256'].items():
            require(sha(name) == digest, 'Waiting queue plan changed')
        for name, digest in read(TRAIN / 'inputs.json')['port_source_manifest'].items():
            require(sha(SOURCE / name) == digest, 'Training source changed: ' + name)
        require([gpu_map().get(i) for i in (4, 5, 6, 7)] == UUIDS, 'Physical GPU UUID mapping changed')

    def no_return():
        require(not any((CTRL / name).exists() for name in TERMINAL), 'Resource return already started')
        require(not (CYCLE / 'resumed-dispatched.json').exists(), 'Borrow cycle already returned')

    def queue_checks():
        for directory in QUEUES:
            plan = read(directory / 'plan.json')
            status = read(directory / 'queue-status.json')
            require(status.get('stage1') == 'COMPLETE' and status.get('roles') ==
                    {'clean': 'WAITING_BORROW_RETURN', 'combo': 'WAITING_BORROW_RETURN'}, 'Queue already left waiting')
            for role in ('clean', 'combo'):
                gate = plan['gates'][role]
                require(gate['kind'] == 'global' and Path(gate['final']) == CTRL / 'final.json'
                        and gate['receipts'] == [str(CTRL / 'rlt-status.json')]
                        and Path(gate['cycle_plan']) == CYCLE / 'plan.json'
                        and gate['cycle_sha256'] == contract['cycle_plan_sha256']
                        and not gate.get('anchors') and short_same(gate['owner'], old['owner']), 'Queue gate differs')
                require(not any((directory / (role + suffix)).exists() for suffix in
                                ('-cutover-bound.json', '-dispatched.json', '-failure.json')), 'Queue cutover already started')
            expected = contract['queue_owners'][str(directory)]
            require(owned(expected), 'Waiting queue owner changed/exited')

    def cleanup_active(graceful=True):
        nonlocal active, popped
        if active is None:
            return
        if graceful and owned(active.root):
            active.signal(active.root, signal.SIGTERM)
            deadline = time.monotonic() + 180
            while owned(active.root) and time.monotonic() < deadline:
                active.scan()
                time.sleep(.5)
        result = active.cleanup()
        managed.extend(active.rows.values())
        atomic(stage / ('cleanup-' + active.phase + '.json'), result)
        if popped is not None:
            popped.wait(timeout=10)
        wait_release(managed, owned)
        active = popped = None
        metadata['child'] = None

    def hold(error=None):
        state('ACTION_REQUIRED_HOLD' if error else 'MAINTENANCE_HOLD', error=error)
        while True:
            if requested[0]:
                return 'abort-return'
            for filename, action in (('abort-return.json', 'abort-return'), ('resume-request.json', 'resume')):
                path = stage / filename
                if not path.is_file():
                    continue
                try:
                    value = read(path)
                    require(isinstance(value, dict), 'Request must be a JSON object')
                except (OSError, ValueError, TypeError):
                    atomic(stage / 'request-error.json', dict(time=time.time(), path=str(path), error=traceback.format_exc()))
                    continue
                if request_valid(value, contract_sha, me) and value['action'] == action:
                    if error and action == 'resume':
                        # A failed attempt must be explicitly cleared by a new request carrying its observed error hash.
                        if value.get('error_sha256') != hashlib.sha256(error.encode()).hexdigest():
                            continue
                    if action == 'resume':
                        try:
                            fixed(); no_return(); queue_checks()
                            require(active is None, 'Cannot relaunch over an active driver')
                            require(not any(row['gpu'] in UUIDS for row in all_gpu_processes()), 'GPU4-7 unexpectedly occupied')
                        except Exception:
                            error = traceback.format_exc()
                            state('ACTION_REQUIRED_HOLD', error=error)
                            time.sleep(2)
                            continue
                    return action
            state(metadata['status'], error=error)
            time.sleep(2)

    def return_rlt(passed, error=None):
        nonlocal return_started
        cleanup_active()
        fixed(); no_return()
        wait_release(managed, owned)
        require(managed, 'Refuse empty cleanup obligation for adopted EXPO')
        atomic(CTRL / 'release.json', dict(cycle_id=CYCLE.name, terminal_status='completed' if passed else 'failed',
               all_workers_stopped=True, managed_processes=managed, physical_gpus=[4, 5, 6, 7],
               maintenance=str(stage), previous_owner=old['owner'], continuation_owner=me))
        atomic(CTRL / 'resume-intent.json', dict(action='resume', time=time.time(), owner=me,
                                               cycle_id=CYCLE.name, maintenance=str(stage)))
        return_started = True
        state('RESTORING_RLT')
        command = [RLT_PY, '-X', 'faulthandler', '-u', '-B', str(CTRL / 'tools/reborrow_resources.py'),
                   'resume', '--release', str(CTRL / 'release.json')]
        with (CTRL / 'resume.log').open('xb') as output:
            subprocess.run(command, cwd=CTRL / 'tools', stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            result = subprocess.run([RLT_PY, '-u', '-B', str(CTRL / 'tools/reborrow_resources.py'), 'status'],
                                    cwd=CTRL / 'tools', capture_output=True, text=True, check=True, timeout=90)
            status = json.loads(result.stdout.splitlines()[-1])
            atomic(CTRL / 'rlt-status.json', status)
            if status.get('all_first_rounds_verified'):
                require(status.get('cycle_id') == CYCLE.name, 'Returned RLT receipt has another cycle')
                state('RLT_RESTORED', formal_passed=passed)
                atomic(CTRL / 'final.json', dict(time=time.time(), owner=me, scope=new_scope,
                    formal_passed=passed, error=error, gpu_released=True, rlt_first_rounds_verified=True,
                    cycle_id=CYCLE.name, train_root=str(TRAIN), maintenance=str(stage), previous_owner=old['owner']))
                return
            time.sleep(20)
        raise TimeoutError('Original RLT returned but four first-round proofs still need attention')

    with (stage / 'handoff.lock').open('a+') as transition_lock:
        fcntl.flock(transition_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (stage / 'handoff-intent.json').exists(), 'Never replay this maintenance handoff')
        fixed(); no_return(); queue_checks()
        current = read(CTRL / 'current.json')
        require(same(current['owner'], old['owner']) and same(current['child'], old['child'])
                and current['scope'] == old_scope and current['status'] == 'DRIVER_RUNNING'
                and owned(old['owner']) and owned(old['child']), 'Live owner/child route changed')
        proof = read(CYCLE / 'rlt-stopped.json')
        require(proof.get('all_original_drivers_stopped') and proof.get('all_original_namespaces_empty')
                and set(proof.get('gpus_released', [])) == {4, 5, 6, 7}, 'Borrow stop proof incomplete')
        old_roster = read(CTRL / 'process-roster-driver-eval10.json')
        require(same(old_roster['root'], old['child']) and old_roster['scope'] == old_scope, 'Original roster route changed')
        atomic(stage / 'handoff-intent.json', dict(time=time.time(), controller=me, contract_sha256=contract_sha,
                                                  old_owner=old['owner'], old_child=old['child']))
        resource_lock = (CTRL / 'resource-owner.lock').open('a+')
        try:
            signal_exact(old['owner'], signal.SIGSTOP); held = True
            wait_stopped(old['owner'])
            current = read(CTRL / 'current.json')
            require(current['status'] == 'DRIVER_RUNNING' and same(current['owner'], old['owner'])
                    and same(current['child'], old['child']) and owned(old['child']), 'Run changed before adoption')
            no_return(); queue_checks()
            # Build the adopted roster before retirement so exact cleanup is always available afterward.
            active = Roster(old['child'], old_scope, 'driver-eval10', stage / 'roster-adopted.json')
            for row in old_roster['registered']:
                active.rows[(row['pid'], row['start_ticks'])] = row
            active.rows[(old['child']['pid'], old['child']['start_ticks'])]['proof'] = 'identity-pinned adopted EXPO root'
            active.scan(); active.write(); rosters.append(active)
            signal_exact(old['owner'], signal.SIGKILL); retired = True; held = False
            deadline = time.monotonic() + 10
            while owned(old['owner']) and time.monotonic() < deadline:
                time.sleep(.05)
            require(not owned(old['owner']), 'Old owner did not retire')
            fcntl.flock(resource_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic(stage / 'adopted.json', dict(time=time.time(), owner=me, old_owner=old['owner'], child=old['child'],
                                              cycle_id=CYCLE.name, queues_unchanged=True))
            state('WAITING_LEARNER_BOUNDARY')
            deadline = time.monotonic() + 7200
            stop_requested_at = None
            while owned(old['child']):
                require(not requested[0], 'Maintenance owner termination requested')
                active.scan()
                if read(TRAIN / 'run/status.json').get('phase') == 'learner_started':
                    signal_exact(old['child'], signal.SIGSTOP)
                    wait_stopped(old['child'])
                    if read(TRAIN / 'run/status.json').get('phase') == 'learner_started':
                        stop_requested_at = time.time()
                        signal_exact(old['child'], signal.SIGTERM)
                        signal_exact(old['child'], signal.SIGCONT)
                        break
                    signal_exact(old['child'], signal.SIGCONT)
                require(time.monotonic() < deadline, 'No learner boundary within two hours')
                time.sleep(.2)
            state('STOPPING_ADOPTED_DRIVER')
            cleanup_active(graceful=True)
            require(stop_requested_at is not None, 'Adopted EXPO exited before the planned stop boundary')
            stopped = read(TRAIN / 'run/stopped.json')
            require(read(TRAIN / 'run/status.json').get('phase') == 'stopping'
                    and stopped.get('time', 0) >= stop_requested_at,
                    'Driver did not produce a fresh StopControl receipt')
            checkpoint = read(TRAIN / 'run/checkpoint.json')
            require(checkpoint.get('finite') is True and set(checkpoint.get('hashes', {})) ==
                    {'base', 'core', 'replay', 'cadence', 'rng', 'progress'}, 'Full checkpoint proof missing')
            require(checkpoint.get('cadence') == stopped.get('cadence'), 'Stopped cadence differs from committed checkpoint')
            atomic(stage / 'pause-complete.json', dict(time=time.time(), checkpoint=checkpoint,
                stopped=stopped, stop_requested_at=stop_requested_at, managed_processes=managed, owner=me))
            action = hold()
            if action == 'abort-return':
                return_rlt(False, 'Maintenance explicitly aborted')
                return 1
            fixed(); no_return(); queue_checks()
            state('RESUMING_EXPO')
            (TRAIN / 'driver-heartbeat').touch()
            require(not (stage / 'driver.log').exists(), 'Resume is one-shot; existing driver log needs review')
            with (stage / 'driver.log').open('xb') as output:
                popped = subprocess.Popen(driver_command(), cwd=SOURCE, env=child_environment(stage, new_scope),
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            anchor = identity(popped.pid)
            active = Roster(anchor, new_scope, PHASE, stage / 'roster-resumed.json')
            active.write(); rosters.append(active)
            state('DRIVER_RUNNING', child=anchor, scope=new_scope, resumed_from=checkpoint)
            sampled = 0.0
            while popped.poll() is None:
                if requested[0]:
                    raise RuntimeError('Maintenance owner termination requested')
                active.scan()
                require(time.time() - (TRAIN / 'driver-heartbeat').stat().st_mtime <= 900, 'EXPO heartbeat stale >15min')
                if time.monotonic() - sampled >= 15:
                    rows = all_gpu_processes()
                    pids = {row['pid'] for row in active.rows.values()}
                    bad = [row for row in rows if row['pid'] in pids and row['gpu'] not in UUIDS]
                    require(not bad, 'Resumed EXPO created context outside physical4-7: ' + json.dumps(bad))
                    atomic(stage / 'gpu-latest.json', dict(time=time.time(), owner=me, child=anchor,
                                                         managed_rows=[row for row in rows if row['pid'] in pids]))
                    state('DRIVER_RUNNING')
                    sampled = time.monotonic()
                time.sleep(2)
            code = popped.returncode
            cleanup_active(graceful=False)
            complete = read(TRAIN / 'run/complete.json') if (TRAIN / 'run/complete.json').is_file() else {}
            passed = (code == 0 and complete.get('ok') is True and complete.get('budget_completed') is True
                      and complete['cadence']['counters']['physical_actions'] == 20000
                      and complete['cadence']['counters']['pending_calls'] == 0)
            return_rlt(passed, None if passed else 'EXPO continuation exited ' + str(code))
            return 0 if passed else 1
        except BaseException:
            error = traceback.format_exc()
            atomic(stage / 'failure.json', dict(time=time.time(), error=error, retired=retired, owner=me))
            if not retired:
                raise
            # Once ownership changed, never silently abandon the inherited cycle.
            if active is not None:
                try:
                    if owned(active.root) and identity(active.root['pid'])['state'] in ('T', 't'):
                        active.signal(active.root, signal.SIGCONT)
                    cleanup_active()
                except BaseException:
                    error += '\nCLEANUP: ' + traceback.format_exc()
            if return_started:
                state('RLT_RETURN_ACTION_REQUIRED', error=error)
                while True:
                    time.sleep(30)
            action = hold(error)
            if action == 'abort-return':
                try:
                    return_rlt(False, error)
                    return 1
                except BaseException:
                    error += '\nABORT RETURN: ' + traceback.format_exc()
                    state('RLT_RETURN_ACTION_REQUIRED' if return_started else 'ACTION_REQUIRED_HOLD', error=error)
                    while True:
                        time.sleep(30)
            # A failed post-retirement transition cannot replay a Popen or mutable stage.
            state('ACTION_REQUIRED_HOLD', error=error + '\nExplicit new reviewed recovery required; no relaunch replay.')
            while True:
                time.sleep(30)
        finally:
            if held and owned(old['owner']):
                signal_exact(old['owner'], signal.SIGCONT)
            resource_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
