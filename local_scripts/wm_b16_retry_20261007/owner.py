"""One B16 formal owner: run training, clean its processes, return GPU4/5 to RLT.

No Ray dashboard polling. Optional resource observations cannot stop training.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid

ROOT = Path('/data/chenyiteng')
UID = 20001
TOKEN = 'OPENDW_SMOKE_OWNER_TOKEN'
PHASE = 'OPENDW_SMOKE_OWNER_PHASE'
MASKS = ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')
GPUS = [4, 5]
VISIBLE = {'actor': [['4']], 'env': [['5']], 'rollout': [['4']]}
SCOPE_KEYS = {'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST', 'PYTHONPATH', 'LD_PRELOAD',
              '__GL_APPLICATION_PROFILE', '__GL_APPLICATION_PROFILE_LOG', 'HOME', 'USER', 'LOGNAME'}
C = H = None

def _pidfd_syscall(number, *arguments):
    """Existing resource_switch/common.py Linux x86_64 LP64 fallback.

    Native syscall IDs: pidfd_send_signal=424, pidfd_open=434. No bare-PID kill.
    """
    import ctypes
    if (sys.platform != 'linux' or platform.machine() != 'x86_64' or
            ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8):
        raise RuntimeError('pidfd native binding unavailable; syscall fallback requires Linux x86_64 LP64')
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)

def pidfd_open(pid):
    native = getattr(os, 'pidfd_open', None)
    if callable(native):
        return native(int(pid), 0)
    import ctypes
    fd = _pidfd_syscall(434, ctypes.c_int(int(pid)), ctypes.c_uint(0))
    try:
        os.set_inheritable(fd, False)
    except Exception:
        os.close(fd)
        raise
    return fd

def pidfd_send(fd, sig):
    native = getattr(signal, 'pidfd_send_signal', None)
    if callable(native):
        return native(int(fd), int(sig), None, 0)
    import ctypes
    return _pidfd_syscall(424, ctypes.c_int(int(fd)), ctypes.c_int(int(sig)),
                          ctypes.c_void_p(None), ctypes.c_uint(0))

def pidfd_probe():
    """Check kernel/API support before borrowing; signal 0 changes no state."""
    fd = pidfd_open(os.getpid())
    try:
        pidfd_send(fd, 0)
    finally:
        os.close(fd)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def owned_path(path, exists=True):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
    assert not path.is_symlink()
    if exists:
        assert path.exists() and path.stat().st_uid == UID
    return path

def load_lifecycle(plan):
    """Load only an explicitly frozen lifecycle implementation, before use."""
    global C, H
    stage = owned_path(plan['lifecycle_path'])
    script = owned_path(plan['lifecycle_module'])
    assert plan['source_sha256'].get(str(script)) == sha(script), 'Unfrozen lifecycle module'
    spec = importlib.util.spec_from_file_location('opendw_multigpu_borrow_return', script)
    C = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = C
    spec.loader.exec_module(C)
    C.install_helper(stage)
    H = C.H

def normalized(cfg):
    value = copy.deepcopy(cfg)
    for path in [('env', 'group_name'), ('actor', 'group_name'), ('rollout', 'group_name'),
                 ('runner', 'logger', 'log_path'), ('runner', 'logger', 'experiment_name'),
                 ('runner', 'per_worker_log_path'), ('env', 'train', 'video_cfg', 'video_base_dir'),
                 ('env', 'eval', 'video_cfg', 'video_base_dir'), ('env', 'eval', 'task_config', 'save_path'),
                 ('algorithm', 'dvac_gradient_weighting', 'output_dir')]:
        parent = value
        for key in path[:-1]:
            parent = parent[key]
        parent.pop(path[-1], None)
    return value

def atomic(path, value):
    H.atomic(path, value)

def record(path, value):
    H.save(path, value)

def proc_env(pid):
    entries = (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    return dict(item.split(b'=', 1) for item in entries if b'=' in item)

class Catalog:
    def __init__(self, path, token):
        self.path, self.token, self.rows = path, token, {}

    def add(self, ident, phase, proof):
        assert ident and ident['uid'] == UID
        if not H.same(ident):
            return
        key = (ident['pid'], ident['start'])
        if key not in self.rows:
            self.rows[key] = dict(ident, phase=phase, proof=proof)
            self.write()

    def write(self):
        atomic(self.path, {'time': H.now(), 'processes': list(self.rows.values())})

    def scan(self):
        for _ in range(2):
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():
                    continue
                ident = H.proc(int(path.name))
                if not ident or ident['uid'] != UID or ident['state'] in ('Z', 'X'):
                    continue
                if (ident['pid'], ident['start']) in self.rows or ident['pid'] == os.getpid():
                    continue
                parents = [r for r in self.rows.values() if r['pid'] == ident['ppid'] and H.same(r)]
                if parents:
                    self.add(ident, parents[0]['phase'], 'exact registered parent')
                    continue
                try:
                    env = proc_env(ident['pid'])
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                if env.get(TOKEN.encode()) == self.token.encode() and H.same(ident):
                    phase = env.get(PHASE.encode(), b'').decode()
                    if phase:
                        self.add(ident, phase, 'unique owner token and phase')

    def live(self, phase=None):
        return [r for r in self.rows.values() if (phase is None or r['phase'] == phase) and H.same(r)]

    def send(self, ident, sig):
        if not H.same(ident):
            return
        try:
            fd = pidfd_open(ident['pid'])
        except ProcessLookupError:
            return
        try:
            if H.same(ident):
                try:
                    pidfd_send(fd, sig)
                except ProcessLookupError:
                    pass
        finally:
            os.close(fd)

def http(url, endpoint='/health', post=False, timeout=5):
    request = urllib.request.Request(url.rstrip('/')+endpoint, data=b'{}' if post else None,
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def add_allowlist(plan):
    path = ROOT/'security/ray-guard/training_allowlist.json'
    owned_path(path)
    with (path.parent/'recovery-allowlist.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = read(path)
        names = {r['namespace'] for r in plan['trials']} | {plan['management_namespace']}
        after = dict(before, namespaces=sorted(set(before['namespaces']) | names))
        record(Path(plan['owner_dir'])/'allowlist-added.json', {'time': H.now(), 'added': sorted(names-set(before['namespaces']))})
        atomic(path, after)

def cleanup(plan, catalog):
    sent = []
    for sig, seconds in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        deadline = time.monotonic() + seconds
        while True:
            catalog.scan()
            targets = catalog.live()
            if not targets:
                return {'time': H.now(), 'signals': sent, 'all_stopped': True}
            for ident in targets:
                catalog.send(ident, sig)
                sent.append({'pid': ident['pid'], 'start': ident['start'], 'signal': int(sig)})
            if time.monotonic() >= deadline:
                break
            time.sleep(1)
    raise RuntimeError('Owned processes did not exit; RLT return remains blocked')


def observe(plan, catalog, phase):
    """Optional minute-level record; only observed wrong-card use is fatal."""
    try:
        catalog.scan()
        rows = catalog.live()
        gpu = H.gpu_processes(list(range(8)))
        value = {'time': H.now(), 'phase': phase, 'processes': rows, 'gpu_processes': gpu}
        with (Path(plan['owner_dir'])/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
    except Exception as exc:
        print('RESOURCE_OBSERVATION_SKIPPED '+type(exc).__name__+': '+str(exc), flush=True)
        return
    owned_pids = {r['pid'] for r in rows}
    wm_pids = {r['pid'] for r in rows if r['phase'].startswith('service_')}
    assert not [r for r in gpu if r['pid'] in owned_pids and r['gpu'] not in GPUS], 'Owned process used an unassigned GPU'
    assert not [r for r in gpu if r['pid'] in wm_pids and r['gpu'] != 5], 'WM used an unassigned GPU'


def wait_training(plan, driver, service, catalog):
    started, observed = time.monotonic(), -60
    while driver.poll() is None:
        if service.poll() is not None:
            raise RuntimeError('WM service exited during training')
        elapsed = time.monotonic()-started
        if elapsed >= plan['trials'][0]['timeout_seconds']:
            raise TimeoutError('Formal training deadline')
        if elapsed-observed >= 60:
            observe(plan, catalog, 'formal')
            try:
                atomic(Path(plan['owner_dir'])/'state.json',
                       {'time': H.now(), 'phase': 'formal', 'elapsed_seconds': elapsed})
            except OSError as exc:
                print('HEARTBEAT_WRITE_SKIPPED '+str(exc), flush=True)
            observed = elapsed
        time.sleep(5)
    return driver.returncode


def run_driver(plan, key):
    owner, repo = Path(plan['owner_dir']), Path(plan['repo'])
    env = read(plan['environment_file'])
    env.update(read(plan['graphics_fragment']))
    row = next(r for r in plan['trials'] if r['key'] == key)
    target = owner/key
    assert os.environ[TOKEN] == plan['token'] and os.environ[PHASE] == key
    assert not any(k in os.environ for k in MASKS)
    os.environ['PYTHONPATH'] = str(repo)+':'+env.get('PYTHONPATH', '')
    os.environ['RAY_ADDRESS'] = plan['ray_address']
    sys.path.insert(0, str(repo))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(4096, hard)), hard))
    record(target/'driver-identity.json', dict(H.proc(os.getpid()), namespace=row['namespace']))
    from rlinf.scheduler import Cluster
    import ray
    Cluster.NAMESPACE = row['namespace']
    original = ray.init
    def scoped_init(*args, **kwargs):
        args = list(args)
        requested = kwargs.get('address', args[0] if args else None)
        assert requested in (None, 'auto', plan['ray_address']), 'Unexpected shared Ray destination'
        if args:
            args[0] = plan['ray_address']
        else:
            kwargs['address'] = plan['ray_address']
        kwargs['namespace'] = row['namespace']
        runtime = kwargs.setdefault('runtime_env', {})
        runtime.setdefault('env_vars', {}).update({TOKEN: plan['token'], PHASE: key})
        for name in SCOPE_KEYS:
            assert os.environ.get(name)
            runtime['env_vars'][name] = os.environ[name]
        result = original(*args, **kwargs)
        job = ray.get_runtime_context().get_job_id()
        job = job.hex() if hasattr(job, 'hex') else str(job)
        receipt = {'time': H.now(), 'namespace': row['namespace'], 'job_id': job, 'driver': H.proc(os.getpid())}
        if not (target/'ray-job.json').exists():
            record(target/'ray-job.json', receipt)
        else:
            assert read(target/'ray-job.json')['job_id'] == job
        return result
    ray.init = scoped_init
    from dataclasses import asdict
    from rlinf.utils.placement import HybridComponentPlacement
    old_placement = HybridComponentPlacement.__init__
    def checked_placement(self, cfg, cluster):
        old_placement(self, cfg, cluster)
        observed = {}
        for name in ('actor', 'env', 'rollout'):
            rows = self.get_strategy(name).get_placement(cluster, True)
            assert self.get_world_size(name) == 1
            assert [r.visible_accelerators for r in rows] == VISIBLE[name]
            observed[name] = [asdict(r) for r in rows]
        if not (target/'verified-placement.json').exists():
            record(target/'verified-placement.json', observed)
    HybridComponentPlacement.__init__ = checked_placement
    signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(SystemExit(128+sig)))
    config = Path(row['config'])
    sys.argv = [str(repo/'examples/embodiment/train_embodied_agent.py'), '--config-path', str(config.parent),
                '--config-name', config.stem, 'hydra.run.dir=.', 'hydra.output_subdir=null', 'hydra.job.chdir=false']
    code = 0
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except BaseException as exc:
        code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        raise
    finally:
        record(target/'driver-finished.json', {'time': H.now(), 'exit_code': code})
        # The owner owns exact namespace/job cleanup, including on driver crashes.
        if ray.is_initialized():
            ray.shutdown()

def owner_main(input_plan):
    assert os.getuid() == UID and socket.gethostname() == 'h100-gpu01'
    pidfd_probe()
    owner, cycle, repo = map(Path, (input_plan['owner_dir'], input_plan['lifecycle_path'], input_plan['repo']))
    cp = C.load_plan(cycle)
    assert cp['physical_gpus'] == GPUS and not (cycle/'rlt-stopped.json').exists()
    for path, digest in input_plan['source_sha256'].items():
        assert sha(owned_path(path)) == digest, 'Source changed: '+path
    owner.mkdir(mode=0o700)
    lock = (owner/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = dict(input_plan, token=uuid.uuid4().hex, uid=UID,
                ray_address=cp['ray_address'], ray_dashboard_url=cp['ray_dashboard_url'],
                management_namespace='opendw_ops_'+owner.name[-40:])
    row, service = plan['trials'][0], plan['services'][0]
    (owner/'formal').mkdir()
    target = owner/'services'/service['key']; target.mkdir(parents=True)
    record(owner/'owner-plan.json', plan)
    record(owner/'owner-identity.json', dict(H.proc(os.getpid()), time=H.now()))
    catalog = Catalog(owner/'process-catalog.json', plan['token'])
    env = read(plan['environment_file'])
    closing = {'value': False, 'launching': False, 'signal': None}
    def terminate(sig, frame):
        closing['signal'] = sig
        if not closing['value'] and not closing['launching']:
            raise RuntimeError('Owner received signal '+str(sig))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    def launch(argv, cwd, child_env, phase, log):
        closing['launching'] = True
        try:
            child_env = dict(child_env, **{TOKEN: plan['token'], PHASE: phase})
            with Path(log).open('x') as stream:
                child = subprocess.Popen(argv, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            catalog.add(H.proc(child.pid), phase, 'exact Popen child')
        finally:
            closing['launching'] = False
        if closing['signal']:
            raise RuntimeError('Termination requested during launch')
        return child
    error = recovery_error = None
    terminal, results = 'failed', []
    try:
        add_allowlist(plan)
        service_env = dict(env, **service.get('environment', {}))
        service_env['CUDA_VISIBLE_DEVICES'] = '5'
        wm = launch(service['argv'], service['cwd'], service_env, 'service_'+service['key'], target/'service.log')
        deadline = time.monotonic()+service.get('startup_seconds', 1200)
        while True:
            assert wm.poll() is None, 'WM exited while loading'
            try:
                health = http(service['url'])
            except (OSError, ValueError):
                health = None
            if health:
                assert health['ok'] and health['is_offloaded'] and health['pid'] == wm.pid
                assert health['physical_gpu'] == 5 and health['wm_batch_size'] == 16
                record(target/'service-cpu-ready.json', health)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('WM startup deadline')
            time.sleep(5)
        observe(plan, catalog, 'service_cpu_ready')
        # Preserve the exact per-card checkpoint and partial-borrow transaction.
        closing['launching'] = True
        try:
            with (cycle/'operation.lock').open('a') as operation:
                fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                C.stop(cycle)
            assert (cycle/'rlt-stopped.json').is_file() and not H.gpu_processes(GPUS)
        finally:
            closing['launching'] = False
        if closing['signal']:
            raise RuntimeError('Termination requested during borrowing')
        env.update(read(plan['graphics_fragment']))
        for name in MASKS:
            env.pop(name, None)
        env.update(PYTHONPATH=str(repo)+':'+env.get('PYTHONPATH', ''), RAY_ADDRESS=plan['ray_address'])
        record(owner/'scope-activated.json', {'physical_gpus': GPUS})
        argv = [plan['python'], '-u', '-B', str(Path(__file__).resolve()), '--plan', str(owner/'owner-plan.json'), 'driver', '--key', 'formal']
        driver = launch(argv, str(repo), env, 'formal', owner/'formal/driver.log')
        code = wait_training(plan, driver, wm, catalog)
        result = {'key': 'formal', 'exit_code': code}
        results.append(result); record(owner/'formal/result.json', result)
        assert code == 0, 'Training process exited with code '+str(code)
        terminal = 'completed'
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
        record(owner/'error.json', error)
    finally:
        closing['value'] = True
        borrowed = (cycle/'rlt-stopped.json').exists()
        try:
            record(owner/'cleanup.json', cleanup(plan, catalog))
            release = {'time': H.now(), 'cycle_id': cycle.name, 'gpus': GPUS, 'terminal_status': terminal,
                       'all_workers_stopped': True, 'managed_processes': list(catalog.rows.values())}
            if borrowed:
                assert not H.gpu_processes(GPUS), 'GPU4/5 still occupied; do not resume RLT'
                receipt = owner/'smoke-release.json'; record(receipt, release)
                with (cycle/'operation.lock').open('a') as operation:
                    fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    returned = H.resume(cycle, receipt)
                record(owner/'rlt-return-dispatched.json', {'time': H.now(), 'result': returned})
            elif (cycle/'clean-old-stop-attempt.json').exists():
                receipt = owner/'partial-smoke-release.json'; record(receipt, release)
                with (cycle/'operation.lock').open('a') as operation:
                    fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    returned = C.recover_partial(cycle, receipt)
                record(owner/'partial-rlt-recovery.json', returned)
        except BaseException as exc:
            recovery_error = {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
            record(owner/'recovery-error.json', recovery_error)
        record(owner/'final.json', {'time': H.now(), 'mode': plan['mode'], 'physical_gpus': GPUS,
               'terminal_status': terminal, 'trials': results, 'error': error, 'recovery_error': recovery_error,
               'rlt_borrowed': borrowed, 'rlt_return_dispatched': (owner/'rlt-return-dispatched.json').exists(),
               'rlt_first_round_verified': False})
    if error or recovery_error:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('action', choices=('owner', 'driver'))
    parser.add_argument('--key')
    args = parser.parse_args()
    plan = read(owned_path(args.plan))
    load_lifecycle(plan)
    if args.action == 'driver':
        assert args.key
        run_driver(plan, args.key)
    else:
        owner_main(plan)

if __name__ == '__main__':
    main()
