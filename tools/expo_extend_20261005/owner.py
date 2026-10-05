"""One durable CPU owner: snapshot 20k, borrow current RLT, resume EXPO to 60k, return RLT."""
import fcntl
import os
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from common import *
sys.path.insert(0, str(SOURCE / 'tools'))
from expo_smoke_owner import Roster, identity, owned, gpu_map
from expo_process import pidfd_open, pidfd_send, pidfd_probe

def gpu_rows():
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=25))
    return [{'index': i, 'pid': int(p.findtext('pid')), 'type': p.findtext('type'), 'memory': p.findtext('used_memory')}
            for i, g in enumerate(root.findall('gpu')) for p in g.findall('processes/process_info')]

def wait_free(managed=(), seconds=120):
    deadline = time.monotonic() + seconds; empty = None
    while time.monotonic() < deadline:
        rows = gpu_rows(); pids = {p['pid'] for p in managed}
        if not any(r['index'] in (4, 5, 6, 7) or r['pid'] in pids for r in rows) and not any(owned(p) for p in managed):
            if empty is None: empty = time.monotonic()
            elif time.monotonic() - empty >= 2: return
        else: empty = None
        time.sleep(1)
    raise RuntimeError('Owned compute/graphics processes have not released the devices')

def exact_signal(row, sig):
    assert owned(row)
    fd = pidfd_open(row['pid'])
    try:
        assert owned(row)
        pidfd_send(fd, sig)
    finally: os.close(fd)

def release_kind(passed, started):
    return 'completed' if passed else 'failed' if started else 'not_started'

def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); pidfd_probe()
    locks = []
    for path in [CONTROL / 'owner.lock', TRAIN / 'resource-owner.lock', ROOT / 'eval10-continuation-20261003/resource-owner.lock']:
        handle = path.open('a+'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    assert not (CONTROL / 'owner.json').exists()
    staged = read(CONTROL / 'staged.json')
    me = identity(os.getpid()); scope = 'expo60k-' + str(uuid.uuid4())
    meta = dict(owner=me, scope=scope, time=time.time(), train=str(TRAIN), control=str(CONTROL),
                physical_gpus=[4, 5, 6, 7], max_physical_actions=60000, status='PREPARING')
    atomic(CONTROL / 'owner.json', meta)
    def state(name, **extra):
        meta.update(status=name, time=time.time(), **extra); atomic(CONTROL / 'current.json', meta)
    def pins():
        for name, digest in staged['files'].items(): assert sha(name) == digest, name
    def command(argv, label, timeout=1800):
        assert not (CONTROL / (label + '-intent.json')).exists()
        atomic(CONTROL / (label + '-intent.json'), {'time': time.time(), 'owner': me, 'argv': argv})
        with (CONTROL / (label + '.log')).open('xb') as f:
            subprocess.run(argv, cwd=TOOLS, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'), stdin=subprocess.DEVNULL,
                           stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    def operation(action):
        command([RLT_PY, '-u', '-B', str(TOOLS / 'resources.py'), action], 'rlt-' + action)
    requested = [False]
    def signal_request(_sig, _frame): requested[0] = True
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, signal_request)
    child = roster = None; managed = []; passed = stopped = started = returned = False; error = None
    def cleanup():
        nonlocal child, roster
        if roster is None: return
        if child.poll() is None and owned(roster.root):
            roster.signal(roster.root, signal.SIGTERM)
            deadline = time.monotonic() + 180
            while child.poll() is None and time.monotonic() < deadline:
                roster.scan(); time.sleep(1)
        result = roster.cleanup(); managed.extend(roster.rows.values())
        child.wait(timeout=10); atomic(CONTROL / 'expo-cleanup.json', result)
        wait_free(managed); child = roster = None
    try:
        pins(); assert [gpu_map().get(i) for i in (4, 5, 6, 7)] == UUIDS
        state('PREPARING_RLT_CHECKPOINTS')
        if (CONTROL / 'preflight-retry.json').is_file():
            retry = read(CONTROL / 'preflight-retry.json')
            assert retry['rlt_untouched'] and not owned(retry['previous_owner'])
            assert not (CONTROL / 'rlt-stop-intent.json').exists()
            assert read(CONTROL / 'replay-ctime-repair.json')['ok']
        else:
            operation('prepare')
            state('MIGRATING_20K_TO_60K')
            command([EXPO_PY, '-u', '-B', str(TOOLS / 'migrate.py')], 'migration')
        assert set(read(CYCLE / 'plan.json')['runs']) == {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
        migration = read(CONTROL / 'migration.json'); assert migration['ok'] and migration['counters_unchanged']
        if requested[0]: raise RuntimeError('Stop requested before borrowing')
        for stage in QUEUES:
            saved = staged['queues'][str(stage)]
            assert sha(stage / 'plan.json') == saved['plan_sha256'] and owned(saved['owner'])
            assert all(v == 'TRAINING' for v in read(stage / 'queue-status.json')['roles'].values())
        # These completed Stage1 queue owners only monitor their running children.
        # Retire only the pinned CPU parent; the frozen RLT helper owns child shutdown.
        for stage in QUEUES:
            saved = staged['queues'][str(stage)]
            assert sha(stage / 'plan.json') == saved['plan_sha256']
            row = saved['owner']; assert owned(row)
            assert all(v == 'TRAINING' for v in read(stage / 'queue-status.json')['roles'].values())
            atomic(stage / 'retired-by-expo60k.json', {'time': time.time(), 'old_owner': row, 'new_owner': me, 'control': str(CONTROL)})
            exact_signal(row, signal.SIGTERM)
            deadline = time.monotonic() + 15
            while owned(row) and time.monotonic() < deadline: time.sleep(.2)
            assert not owned(row), 'CPU queue owner did not retire'
            lock = (stage / 'owner.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
            atomic(stage / 'active-continuation.json', {'control': str(CONTROL), 'current': str(CONTROL / 'current.json'), 'owner': me, 'cycle': str(CYCLE)})
            atomic(stage / 'queue-status.json', {'time': time.time(), 'stage1': 'COMPLETE', 'roles': {'clean': 'PAUSING_FOR_EXPO', 'combo': 'PAUSING_FOR_EXPO'}, 'control': str(CONTROL), 'owner': me})
        state('STOPPING_RLT'); operation('stop')
        proof = read(CYCLE / 'rlt-stopped.json')
        assert proof['all_original_drivers_stopped'] and proof['all_original_namespaces_empty'] and set(proof['gpus_released']) == {4, 5, 6, 7}
        stopped = True; wait_free()
        pins(); assert sha(TRAIN / 'inputs.json') == migration['new_inputs_sha256']
        for p, digest in read(TRAIN / 'inputs.json')['port_source_manifest'].items(): assert sha(SOURCE / p) == digest
        for stage in QUEUES:
            atomic(stage / 'queue-status.json', {'time': time.time(), 'stage1': 'COMPLETE', 'roles': {'clean': 'WAITING_EXPO_60K', 'combo': 'WAITING_EXPO_60K'}, 'control': str(CONTROL), 'owner': me})
        if requested[0]: raise RuntimeError('Stop requested after borrowing')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(UUIDS), HOME='/home/chenyiteng',
                   PYTHONPATH=str(GRAPHICS / 'bootstrap') + ':' + str(SOURCE) + ':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
                   LD_LIBRARY_PATH='/home/chenyiteng/tools/cuda-12.9/lib64', VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                   PYTHONFAULTHANDLER='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', TMPDIR=str(CONTROL / 'tmp'),
                   EXPO_SMOKE_OWNER_SCOPE=scope, EXPO_SMOKE_OWNER_PHASE='driver60k',
                   RLINF_EXPO_GPU_SCOPE_MANIFEST=str(CONTROL / 'scope.json'), __GL_APPLICATION_PROFILE='1')
        env.pop('DISPLAY', None)
        argv = [EXPO_PY, '-X', 'faulthandler', '-u', '-B', str(SOURCE / 'examples/embodiment/train_expo_formal.py'),
                '--inputs', str(TRAIN / 'inputs.json'), '--run', str(TRAIN / 'run'), '--max-physical-actions', '60000',
                '--enable-evaluation', '--resume', str(TRAIN / 'run/checkpoint-latest.pt')]
        (TRAIN / 'driver-heartbeat').touch()
        with (CONTROL / 'driver.log').open('xb') as f:
            child = subprocess.Popen(argv, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        started = True
        anchor = identity(child.pid); roster = Roster(anchor, scope, 'driver60k', CONTROL / 'expo-roster.json'); roster.write()
        state('EXPO_RUNNING', child=anchor, argv=argv, checkpoint_start={'actions': 20000, 'episodes': 128, 'calls': 454})
        atomic(TRAIN / 'active-continuation.json', {'control': str(CONTROL), 'current': str(CONTROL / 'current.json'), 'owner': me, 'child': anchor, 'cycle': str(CYCLE), 'source': str(SOURCE), 'budget': 60000})
        sampled = 0
        while child.poll() is None:
            if requested[0]: raise RuntimeError('EXPO owner stop requested')
            roster.scan()
            if time.time() - (TRAIN / 'driver-heartbeat').stat().st_mtime > 900: raise TimeoutError('EXPO heartbeat stale >15min')
            if time.monotonic() - sampled > 15:
                rows = gpu_rows(); pids = {r['pid'] for r in roster.rows.values()}
                if any(r['pid'] in pids and r['index'] not in (4, 5, 6, 7) for r in rows): raise RuntimeError('EXPO compute/graphics escaped GPUs4-7')
                state('EXPO_RUNNING', gpu_processes=[r for r in rows if r['pid'] in pids]); sampled = time.monotonic()
            time.sleep(2)
        code = child.returncode; cleanup()
        atomic(CONTROL / 'expo-exit.json', {'time': time.time(), 'returncode': code})
        assert code == 0, 'EXPO exited ' + str(code)
        done = read(TRAIN / 'run/complete.json')
        assert done['ok'] and done['budget_completed'] and done['cadence']['counters']['physical_actions'] == 60000
        assert done['cadence']['counters']['pending_calls'] == 0
        passed = True
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        try:
            cleanup()
            if not stopped and (CYCLE / 'rlt-stopped.json').is_file():
                proof = read(CYCLE / 'rlt-stopped.json')
                stopped = (proof['cycle_id'] == CYCLE.name and proof['all_original_drivers_stopped']
                           and proof['all_original_namespaces_empty'] and set(proof['gpus_released']) == {4, 5, 6, 7})
            if stopped:
                wait_free(managed); pins()
                atomic(CONTROL / 'release.json', {'cycle_id': CYCLE.name, 'terminal_status': release_kind(passed, started),
                       'all_workers_stopped': True, 'managed_processes': managed, 'physical_gpus': [4, 5, 6, 7]})
                state('RESTORING_RLT', error=error); operation('resume')
                for stage in QUEUES:
                    atomic(stage / 'queue-status.json', {'time': time.time(), 'stage1': 'COMPLETE', 'roles': {'clean': 'RESUMING', 'combo': 'RESUMING'}, 'control': str(CONTROL), 'owner': me})
                deadline = time.monotonic() + 1800
                while time.monotonic() < deadline:
                    with (CONTROL / 'rlt-status.log').open('w') as f:
                        subprocess.run([RLT_PY, '-u', '-B', str(TOOLS / 'resources.py'), 'status'], cwd=TOOLS,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=f, stderr=subprocess.STDOUT, check=True, timeout=90)
                    if read(CONTROL / 'rlt-status.json')['all_first_rounds_verified']:
                        returned = True; break
                    time.sleep(20)
                assert returned, 'RLT dispatched, but real resumed learning not yet verified'
                state('RLT_RESTORED', expo_completed=passed, error=error)
                for stage in QUEUES:
                    atomic(stage / 'queue-status.json', {'time': time.time(), 'stage1': 'COMPLETE', 'roles': {'clean': 'TRAINING', 'combo': 'TRAINING'}, 'control': str(CONTROL), 'rlt_status': str(CONTROL / 'rlt-status.json')})
            elif (CONTROL / 'rlt-stop-intent.json').exists():
                raise RuntimeError('Partial RLT stop: inspect exact stop receipt before any further dispatch')
            else: state('FAILED_BEFORE_BORROW', error=error)
        except BaseException as exc:
            error = {'type': 'CLEANUP_OR_RETURN_NEEDS_ACTION', 'message': str(exc), 'previous': error, 'traceback': traceback.format_exc()}
            state('ACTION_REQUIRED', error=error)
        atomic(CONTROL / 'final.json', {'time': time.time(), 'owner': me, 'expo_completed': passed, 'error': error,
               'gpus_released': stopped and child is None, 'rlt_first_rounds_verified': returned, 'budget': 60000})
    return 0 if passed and returned else 1

if __name__ == '__main__': raise SystemExit(main())
