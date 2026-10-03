"""New resource cycle; resume the original EXPO run with only eval25 -> eval10."""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

TRAIN_ROOT = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002')
ROOT = TRAIN_ROOT.parent / 'eval10-continuation-20261003'
SOURCE = TRAIN_ROOT / 'source'
TOOLS = ROOT / 'tools'
CYCLE = ROOT / 'rlt-cycle-expo-eval10-20261003-v1'
PREVIOUS = TRAIN_ROOT / 'rlt-cycle-expo-turn-switch-repair-20261002-v1'
RLT_RUNS = Path('/data/chenyiteng/results/rlinf-rlt')
RLT_PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
EXPO_PY = '/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python'
UUIDS = ['GPU-a0a252d6-828d-29e1-1fd2-65187f573f4d', 'GPU-2cd891ea-180d-da39-6419-2d7033f8b21b',
         'GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4', 'GPU-3c6321c1-3e58-c071-3867-533391152fe7']
QUEUES = {Path('/data/chenyiteng/deployment-20261002/rlt-next6-' + task)
          for task in ('place_object_stand', 'move_playingcard_away')}
PAYLOAD = {'base', 'core', 'replay', 'cadence', 'rng', 'progress'}
PHASE = 'driver-eval10'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path); before = path.stat(); value = hashlib.sha256()
    with path.open('rb') as stream:
        require((os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino)
                == (before.st_dev, before.st_ino), 'Source replaced before hash')
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Source changed while hashing: ' + str(path))
    return value.hexdigest()


def pin_files(pins):
    require(isinstance(pins, dict) and pins, 'Empty source pins')
    account = Path('/data/chenyiteng').resolve()
    for name, digest in pins.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.stat().st_uid == os.getuid() and path.resolve().is_relative_to(account),
                'Invalid pinned account file: ' + name)
        require(sha(path) == digest, 'Pinned source differs: ' + name)


def anchor(row):
    start = row.get('start_ticks', row.get('start'))
    require(row.get('uid') == 20001 and type(row.get('pid')) is int and row['pid'] > 1
            and type(start) is int and start > 0, 'Missing exact process identity')
    return row['pid'], row['uid'], start


def exact_live(row, identity):
    try:
        actual = identity(row['pid'])
        return (anchor(actual) == anchor(row) and actual['state'] not in ('Z', 'X')
                and ('boot_id' not in row or row['boot_id'] == actual['boot_id']))
    except (OSError, ValueError, KeyError):
        return False


def owned_path(path, parent, *, directory=False):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink()
            and path.resolve(strict=True).is_relative_to(Path(parent).resolve(strict=True))
            and path.stat().st_uid == os.getuid()
            and (path.is_dir() if directory else path.is_file()),
            'Unexpected checkpoint-wait path/owner: ' + str(path))
    return path


def wait_rlt_checkpoints(identity, *, timeout=7200, sleep=time.sleep,
                         monotonic=time.monotonic, report=None):
    """Readiness only: keep returned RLT running until each saves a fresh CP."""
    plan_path = owned_path(PREVIOUS / 'plan.json', PREVIOUS)
    dispatch_path = owned_path(PREVIOUS / 'resumed-dispatched.json', PREVIOUS)
    plan = read(plan_path); dispatched = read(dispatch_path)
    pins = {plan_path: sha(plan_path), dispatch_path: sha(dispatch_path)}
    keys = {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
    require(plan.get('cycle_id') == PREVIOUS.name and plan.get('host') == 'h100-gpu02'
            and set(plan.get('runs', {})) == keys and dispatched.get('cycle_id') == PREVIOUS.name
            and dispatched.get('runs') == {key: row['new_run'] for key, row in plan['runs'].items()},
            'Previous returned-run plan/dispatch differs')
    runs = {}
    for key, row in plan['runs'].items():
        require(row['gpus'] == [int(key[3:])], 'Returned GPU mapping differs: ' + key)
        run = owned_path(Path(row['new_run']), RLT_RUNS, directory=True)
        require(run.parent.resolve() == RLT_RUNS.resolve(), 'Returned run is outside exact results root')
        runtime = owned_path(run / 'runtime', run, directory=True)
        launched_path = owned_path(PREVIOUS / (key + '-launched.json'), PREVIOUS)
        launched = read(launched_path)
        require(launched.get('run') == str(run) and launched.get('namespace') == row['namespace'],
                'Returned launch route differs: ' + key)
        anchor(launched['identity']); pins[launched_path] = sha(launched_path)
        runs[key] = (run, runtime, launched['identity'], row['namespace'])
    deadline = monotonic() + timeout
    while True:
        for path, digest in pins.items():
            owned_path(path, PREVIOUS)
            require(sha(path) == digest, 'Previous return receipt changed during checkpoint wait')
        observed = {}
        for key, (run, runtime, launched, namespace) in runs.items():
            owned_path(run, RLT_RUNS, directory=True); owned_path(runtime, run, directory=True)
            current = read(owned_path(runtime / 'driver-identity.json', runtime))
            require(anchor(current) == anchor(launched) and current.get('namespace') == namespace
                    and exact_live(launched, identity) and exact_live(current, identity),
                    'Returned RLT driver died/changed while awaiting checkpoint: ' + key)
            require(not any((runtime / name).exists() for name in
                    ('finished.json', 'exit_code.txt', 'cleanup-targets.json')),
                    'Returned RLT entered termination/cleanup: ' + key)
            roots = []
            for parts in ((run.name, 'checkpoints'), ('checkpoints',)):
                path = run
                for part in parts:
                    path = path / part
                    if not path.exists() and not path.is_symlink():
                        break
                    owned_path(path, run, directory=True)
                else:
                    roots.append(path)
            require(len(roots) <= 1, 'Ambiguous checkpoint roots for returned RLT: ' + key)
            checkpoint = None
            if roots:
                candidates = sorted((path for path in roots[0].iterdir()
                    if re.fullmatch(r'global_step_\d+', path.name)),
                    key=lambda path: int(path.name.rsplit('_', 1)[1]), reverse=True)
                for path in candidates:
                    owned_path(path, run, directory=True)
                    state_dir = path
                    for part in ('actor', 'sac_components', 'rlt_trainer_state'):
                        state_dir = state_dir / part
                        if not state_dir.exists() and not state_dir.is_symlink():
                            break
                        owned_path(state_dir, run, directory=True)
                    else:
                        marker = state_dir / 'complete.json'
                        if marker.exists() or marker.is_symlink():
                            if read(owned_path(marker, run)).get('complete') is True:
                                checkpoint = str(path); break
            observed[key] = dict(run=str(run), driver=launched,
                checkpoint_root=str(roots[0]) if roots else None, checkpoint=checkpoint)
        if report is not None:
            report(observed)
        if all(row['checkpoint'] is not None for row in observed.values()):
            # Payload/contract validation remains exclusively in bridge.prepare.
            return observed
        if monotonic() >= deadline:
            raise TimeoutError('Returned RLT did not save four fresh complete checkpoints within two hours')
        sleep(min(30, max(0, deadline - monotonic())))


def validate_launch(contract):
    require(contract.get('version') == 1 and contract.get('ok') is True
            and contract.get('root') == str(ROOT.resolve())
            and contract.get('train_root') == str(TRAIN_ROOT.resolve())
            and contract.get('previous_cycle') == str(PREVIOUS.resolve()), 'Launch identity differs')
    required = {str(TOOLS / name) for name in ('reborrow_owner.py', 'reborrow_resources.py')}
    required.add(str(ROOT / 'migrate_eval10.py'))
    required |= {str(SOURCE / 'tools' / name) for name in
                 ('expo_formal_owner.py', 'expo_formal_resources.py', 'expo_rlt_switch.py',
                  'expo_smoke_owner.py', 'expo_process.py')}
    require(required <= set(contract['source_sha256']), 'Owner/migration/helper source pin missing')
    require(len(contract.get('inputs_before_sha256', '')) == 64, 'Original inputs SHA missing')


def validate_rebind(ready, plan, owner):
    require(ready.get('version') == 1 and ready.get('ok') is True
            and ready.get('cycle_id') == CYCLE.name and anchor(ready['owner']) == anchor(owner),
            'Queue rebind receipt is for another owner/cycle')
    rows = ready.get('queues', [])
    require(len(rows) == 2 and {Path(row['stage']) for row in rows} == QUEUES, 'Expected two exact queues')
    require(plan.get('cycle_id') == CYCLE.name and plan.get('predecessor_cycle') == str(PREVIOUS),
            'Prepared cycle does not follow the returned original cycle')


def check_queues(ready, owner, identity):
    cycle_plan = read(CYCLE / 'plan.json')
    validate_rebind(ready, cycle_plan, owner)
    require(ready['cycle_plan_sha256'] == sha(CYCLE / 'plan.json'), 'Prepared cycle plan changed')
    for row in ready['queues']:
        stage = Path(row['stage'])
        require(not exact_live(row['old_owner'], identity), 'Old cached-plan queue remains alive')
        require(exact_live(row['new_owner'], identity), 'Rebound queue owner is not alive')
        require(row['plan_sha256'] == sha(stage / 'plan.json'), 'Queue plan hash differs')
        pin_files(row['source_sha256'])
        queue = read(stage / 'plan.json')
        status = read(stage / 'queue-status.json')
        require(status.get('stage1') == 'COMPLETE'
                and all(status['roles'][role] == 'WAITING_BORROW_RETURN' for role in ('clean', 'combo')),
                'Rebound queue is not waiting after completed Stage1')
        stage1 = read(stage / 'stage1-complete.json')
        require(Path(stage1['weights']).is_file() and Path(stage1['weights']).stat().st_size == stage1['bytes'],
                'Completed Stage1 weights are missing/changed')
        for role in ('clean', 'combo'):
            for suffix in ('cutover-bound.json', 'dispatched.json', 'first-round.json'):
                require(not (stage / (role + '-' + suffix)).exists(), 'Queue already started a cutover')
            gate = queue['gates'][role]
            key = 'gpu' + str(gate['gpu']); returned = cycle_plan['runs'][key]
            require(gate['kind'] == 'global' and Path(gate['cycle_plan']).resolve() == (CYCLE / 'plan.json').resolve()
                    and gate['cycle_sha256'] == ready['cycle_plan_sha256']
                    and Path(gate['final']).resolve() == (ROOT / 'final.json').resolve()
                    and [Path(path).resolve() for path in gate['receipts']] == [(ROOT / 'rlt-status.json').resolve()]
                    and anchor(gate['owner']) == anchor(owner), 'Queue gate does not await this EXPO owner')
            targets = [target for target in queue['old_runs'] if target['gpus'] == [gate['gpu']]]
            require(len(targets) == 1 and targets[0]['run'] == returned['new_run']
                    and targets[0]['namespace'] == returned['namespace'], 'Queue returned-run route differs')


def wait_gpu_release(gpu_processes, *, timeout=90, sleep=time.sleep, monotonic=time.monotonic):
    """Allow CUDA teardown lag; require two empty observations >=2 seconds apart."""
    require(timeout >= 60, 'GPU teardown settle window must be at least 60 seconds')
    deadline = monotonic() + timeout; empty_since = None
    while True:
        now = monotonic()
        rows = [row for row in gpu_processes() if row['gpu'] in UUIDS]
        if not rows:
            if empty_since is None:
                empty_since = now
            elif now - empty_since >= 2:
                return
        else:
            empty_since = None
        if now >= deadline:
            raise TimeoutError('GPU4-7 remains occupied after CUDA teardown settle window')
        sleep(1)


def driver_command():
    return [EXPO_PY, '-X', 'faulthandler', '-u', '-B', str(SOURCE / 'examples/embodiment/train_expo_formal.py'),
            '--inputs', str(TRAIN_ROOT / 'inputs.json'), '--run', str(TRAIN_ROOT / 'run'),
            '--max-physical-actions', '20000', '--enable-evaluation',
            '--resume', str(TRAIN_ROOT / 'run/checkpoint-latest.pt')]


def validate_migration(result, original_sha):
    require(result.get('ok') is True and result.get('old_inputs_sha256') == original_sha,
            'Migration failed or starts from different inputs')
    before = result.get('old_payload_hashes')
    require(isinstance(before, dict) and set(before) == PAYLOAD
            and before == result.get('new_payload_hashes'), 'Migration altered training state')
    require(result.get('save_policy_changed') is False and result.get('replay_path_identity_changed') is False,
            'Migration changed checkpoint policy/replay identity')


def release_terminal_status(passed, managed, driver_started):
    if passed:
        return 'completed'
    return 'not_started' if not managed and not driver_started else 'failed'


class Terminated(RuntimeError):
    pass


def main():
    require(os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02', 'Wrong host/account')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Owner must be CPU-only')
    sys.path.insert(0, str(SOURCE / 'tools'))
    from expo_smoke_owner import Roster, atomic, identity, owned, gpu_map, gpu_processes
    from expo_process import pidfd_probe
    pidfd_probe(); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    contract = read(ROOT / 'launch-contract.json'); validate_launch(contract); pin_files(contract['source_sha256'])
    require(sha(TRAIN_ROOT / 'inputs.json') == contract['inputs_before_sha256'], 'Original inputs changed')
    with (ROOT / 'resource-owner.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((ROOT / name).exists() for name in ('owner.json', 'final.json', 'prepare-intent.json')),
                'New owner was already dispatched')
        me = identity(os.getpid()); scope = 'expo-eval10-reborrow-' + str(uuid.uuid4())
        metadata = dict(owner=me, scope=scope, status='PREPARING', time=time.time(), cycle_id=CYCLE.name,
                        physical_gpus=[4, 5, 6, 7], train_root=str(TRAIN_ROOT),
                        evaluation_every_episodes=10, evaluation_episodes=20)
        atomic(ROOT / 'owner.json', metadata); atomic(ROOT / 'current.json', metadata)
        requested = [False]; closing = [False]; critical = [False]

        def terminate(sig, _frame):
            requested[0] = True
            if not closing[0] and not critical[0]:
                raise Terminated('Owner signal ' + str(sig))

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, terminate)

        def state(name, **more):
            metadata.update(status=name, time=time.time(), **more)
            atomic(ROOT / 'current.json', metadata)

        def fixed_source():
            pin_files(contract['source_sha256'])

        def operation(action, release=None):
            require(action in ('prepare', 'stop', 'resume'), 'Unsupported resource mutation')
            path = ROOT / (action + '-intent.json')
            require(not path.exists(), 'Refusing repeated ' + action)
            fixed_source(); atomic(path, dict(action=action, time=time.time(), owner=me, cycle_id=CYCLE.name))
            command = [RLT_PY, '-X', 'faulthandler', '-u', '-B', str(TOOLS / 'reborrow_resources.py'), action]
            if release is not None:
                command += ['--release', str(release)]
            critical[0] = True
            try:
                with (ROOT / (action + '.log')).open('xb') as output:
                    subprocess.run(command, cwd=TOOLS, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
            finally:
                critical[0] = False

        child = roster = None; managed = []; stopped = passed = released = False; error = None; returned = None
        driver_started = False

        def cleanup_child(graceful=False):
            nonlocal child, roster
            if roster is None:
                return
            if graceful and child.poll() is None and owned(roster.root):
                roster.signal(roster.root, signal.SIGTERM); deadline = time.monotonic() + 180
                while child.poll() is None and time.monotonic() < deadline:
                    roster.scan(); time.sleep(1)
            result = roster.cleanup(); managed.extend(roster.rows.values())
            atomic(ROOT / ('cleanup-' + PHASE + '.json'), result)
            child.wait(timeout=10); wait_gpu_release(gpu_processes)
            child = roster = None

        try:
            require([gpu_map().get(i) for i in (4, 5, 6, 7)] == UUIDS, 'GPU UUID mapping differs')
            state('WAITING_PREVIOUS_RLT_RETURN')
            deadline = time.monotonic() + 1800
            while True:
                old_current = read(TRAIN_ROOT / 'current.json')
                final_path = TRAIN_ROOT / 'final.json'
                if final_path.is_file() and old_current.get('status') == 'RLT_RESTORED':
                    old_final = read(final_path)
                    if (old_final.get('gpu_released') is True and old_final.get('rlt_first_rounds_verified') is True
                            and not exact_live(old_current['owner'], identity)):
                        break
                if requested[0]:
                    raise Terminated('Stopped while awaiting previous owner return')
                if time.monotonic() > deadline:
                    raise TimeoutError('Previous owner return was not completed within 30 minutes')
                time.sleep(2)
            state('WAITING_FOR_RLT_CHECKPOINTS')
            wait_rlt_checkpoints(identity, report=lambda rows: state(
                'WAITING_FOR_RLT_CHECKPOINTS', rlt_checkpoint_wait=rows))
            operation('prepare')
            state('PREPARED_WAITING_QUEUE_REBIND', cycle_plan=str(CYCLE / 'plan.json'),
                  cycle_plan_sha256=sha(CYCLE / 'plan.json'))
            deadline = time.monotonic() + 3600
            ready_path = ROOT / 'queue-rebind-ready.json'
            while not ready_path.is_file():
                if requested[0]:
                    raise Terminated('Stopped while awaiting queue rebind')
                if time.monotonic() > deadline:
                    raise TimeoutError('Queue rebind did not arrive within one hour')
                time.sleep(2)
            ready = read(ready_path); check_queues(ready, me, identity); fixed_source()
            require(sha(TRAIN_ROOT / 'inputs.json') == contract['inputs_before_sha256'], 'Inputs changed before stop')
            state('STOPPING_RLT'); operation('stop')
            proof = read(CYCLE / 'rlt-stopped.json')
            require(proof.get('all_original_drivers_stopped') is True
                    and proof.get('all_original_namespaces_empty') is True
                    and set(proof.get('gpus_released', [])) == {4, 5, 6, 7}, 'New RLT stop proof incomplete')
            stopped = True; wait_gpu_release(gpu_processes)
            if requested[0]:
                raise Terminated('Stopped after RLT borrow')
            state('MIGRATING_EVALUATION_CONTRACT')
            critical[0] = True
            try:
                with (ROOT / 'migration.log').open('xb') as output:
                    subprocess.run([EXPO_PY, '-u', '-B', str(ROOT / 'migrate_eval10.py')], cwd=ROOT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
            finally:
                critical[0] = False
            result = read(ROOT / 'migration-receipt.json'); validate_migration(result, contract['inputs_before_sha256'])
            require(sha(TRAIN_ROOT / 'inputs.json') == result['new_inputs_sha256'], 'Migrated inputs SHA differs')
            require(sha(TRAIN_ROOT / 'run/checkpoint-latest.pt') == result['new_checkpoint_sha256'], 'Migrated checkpoint SHA differs')
            inputs = read(TRAIN_ROOT / 'inputs.json')
            require(inputs['evaluation']['every_episodes'] == 10 and inputs['evaluation']['episodes'] == 20,
                    'Evaluation cadence/sample count differs')
            for name, fingerprint in inputs['port_source_manifest'].items():
                require(sha(SOURCE / name) == fingerprint, 'Migrated training source differs: ' + name)
            fixed_source(); wait_gpu_release(gpu_processes)
            if requested[0]:
                raise Terminated('Stopped before resumed driver')
            temporary = ROOT / 'tmp'; temporary.mkdir(mode=0o700, exist_ok=True)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(UUIDS),
                PYTHONPATH=str(SOURCE) + ':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
                LD_LIBRARY_PATH='/home/chenyiteng/tools/cuda-12.9/lib64', VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                PYTHONFAULTHANDLER='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', TMPDIR=str(temporary),
                EXPO_SMOKE_OWNER_SCOPE=scope, EXPO_SMOKE_OWNER_PHASE=PHASE)
            command = driver_command(); (TRAIN_ROOT / 'driver-heartbeat').touch()
            critical[0] = True
            try:
                with (ROOT / (PHASE + '.log')).open('xb') as output:
                    child = subprocess.Popen(command, cwd=SOURCE, env=environment, stdin=subprocess.DEVNULL,
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    driver_started = True
                    child_anchor = identity(child.pid)
                    roster = Roster(child_anchor, scope, PHASE, ROOT / ('process-roster-' + PHASE + '.json')); roster.write()
            finally:
                critical[0] = False
            state('DRIVER_RUNNING', child=child_anchor, command=command, migration_receipt=str(ROOT / 'migration-receipt.json'))
            sampled = time.monotonic()
            while child.poll() is None:
                if requested[0]:
                    raise Terminated('Requested during resumed driver')
                roster.scan()
                if time.time() - (TRAIN_ROOT / 'driver-heartbeat').stat().st_mtime > 900:
                    raise TimeoutError('Training driver heartbeat stale >15min')
                if time.monotonic() - sampled > 30:
                    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                        '--format=csv,noheader'], text=True, timeout=20)
                    with (ROOT / 'gpu-samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(time=time.time(), phase=PHASE, gpu=output)) + '\n')
                    sampled = time.monotonic()
                time.sleep(2)
            code = child.returncode; cleanup_child()
            atomic(ROOT / (PHASE + '-exit.json'), dict(time=time.time(), returncode=code, gpu_released=True))
            require(code == 0, 'EXPO continuation exited ' + str(code))
            complete = read(TRAIN_ROOT / 'run/complete.json')
            require(complete.get('ok') is True and complete.get('budget_completed') is True
                    and complete['cadence']['counters']['physical_actions'] == 20000
                    and complete['cadence']['counters']['pending_calls'] == 0, 'Training budget not completed')
            passed = True
        except BaseException as failure:
            error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
        finally:
            closing[0] = True
            try:
                if roster is not None:
                    state('CLEANING_EXPO'); cleanup_child(graceful=True)
                if not stopped and (CYCLE / 'rlt-stopped.json').is_file():
                    proof = read(CYCLE / 'rlt-stopped.json')
                    stopped = (proof.get('all_original_drivers_stopped') is True
                        and proof.get('all_original_namespaces_empty') is True
                        and set(proof.get('gpus_released', [])) == {4, 5, 6, 7})
                if stopped:
                    wait_gpu_release(gpu_processes); released = True; fixed_source()
                    require(not (ROOT / 'release.json').exists(), 'New cycle release already exists')
                    atomic(ROOT / 'release.json', dict(cycle_id=CYCLE.name,
                        terminal_status=release_terminal_status(passed, managed, driver_started),
                        all_workers_stopped=True, managed_processes=managed, physical_gpus=[4, 5, 6, 7]))
                    state('RESTORING_RLT'); operation('resume', ROOT / 'release.json')
                    deadline = time.monotonic() + 1800
                    while time.monotonic() < deadline:
                        with (ROOT / 'rlt-status.log').open('w') as output:
                            subprocess.run([RLT_PY, '-u', '-B', str(TOOLS / 'reborrow_resources.py'), 'status'], cwd=TOOLS,
                                stdout=output, stderr=subprocess.STDOUT, check=True, timeout=90)
                        result = json.loads((ROOT / 'rlt-status.log').read_text().splitlines()[-1]); atomic(ROOT / 'rlt-status.json', result)
                        if result.get('all_first_rounds_verified'):
                            returned = result; break
                        time.sleep(20)
                    require(returned is not None, 'RLT dispatched; first-round verification needs attention')
                    state('RLT_RESTORED', formal_passed=passed)
                elif (ROOT / 'stop-intent.json').exists():
                    raise RuntimeError('Partial RLT stop; inspect new-cycle stop receipt before recovery')
                else:
                    state('FAILED_BEFORE_BORROW', error=error)
            except BaseException as failure:
                error = dict(type='CLEANUP_OR_RLT_RETURN_NEEDS_ACTION', message=str(failure), previous=error,
                             traceback=traceback.format_exc()); state('ACTION_REQUIRED')
            atomic(ROOT / 'final.json', dict(time=time.time(), owner=me, scope=scope, formal_passed=passed, error=error,
                gpu_released=released, rlt_first_rounds_verified=bool(returned), cycle_id=CYCLE.name, train_root=str(TRAIN_ROOT)))
        return 0 if passed and returned and error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
