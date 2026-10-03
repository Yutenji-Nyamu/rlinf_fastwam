"""Bounded SZ3 GPU4-7 N64/R8 smoke owner and shared Ray driver.

Two independent CPU-first WM services run on GPU6/7. Actor and rollout each
have two ranks on GPU4/5. The separately reviewed lifecycle module owns all
four-card borrowing/return decisions; return occurs after the complete suite.
No private Ray instance is started. Process signals require pinned identities.
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
import re
import resource
import runpy
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import uuid

ROOT = Path('/data/chenyiteng')
UID = 20001
TOKEN = 'OPENDW_SMOKE_OWNER_TOKEN'
PHASE = 'OPENDW_SMOKE_OWNER_PHASE'
MASKS = ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')
C = None
H = None
GPUS = [4, 5, 6, 7]
PLACEMENT = {'actor': '4,5', 'env': '6,7', 'rollout': '4,5'}
VISIBLE = {'actor': [['4'], ['5']], 'env': [['6'], ['7']], 'rollout': [['4'], ['5']]}


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
    C.load_plan(stage)


def validate_config(cfg, row, plan, owner):
    """Pure schema checks for this exact one-iteration parallel smoke."""
    episode_steps = row['episode_steps']
    assert episode_steps in (32, 384)
    global_batch = 512 if episode_steps == 32 else 2048
    assert cfg['cluster']['component_placement'] == PLACEMENT
    runner, env, actor, algorithm = cfg['runner'], cfg['env']['train'], cfg['actor'], cfg['algorithm']
    assert runner['max_steps'] == runner['max_epochs'] == 1
    assert runner['val_check_interval'] == -1 and runner['save_interval'] == 1
    assert runner.get('only_eval') is False
    assert runner.get('resume_dir') is None and runner.get('ckpt_path') is None
    assert env['env_type'] == 'opendw_robotwin' and env['task_name'] == 'adjust_bottle'
    assert row['num_envs'] == env['total_num_envs'] == 64
    assert env['rollout_epoch'] == 8 and env['group_size'] == algorithm['group_size'] == 8
    assert env['chunk'] == 32 and env['max_episode_steps'] == env['max_steps_per_rollout_epoch'] == episode_steps
    assert env['frame_stride'] == 4 and env['auto_reset'] is False and env['ignore_terminations'] is False
    assert env['use_rel_reward'] is True and env['reward_coef'] == 1.0
    assert env['success_reward_threshold'] == 0.9
    assert algorithm['adv_type'] == 'grpo' and algorithm['reward_type'] == 'chunk_level'
    assert algorithm['filter_rewards'] is True
    assert algorithm['rewards_lower_bound'] == 0.1 and algorithm['rewards_upper_bound'] == 0.9
    assert algorithm['update_epoch'] == 2
    assert actor['global_batch_size'] == global_batch and actor['micro_batch_size'] == 8
    assert actor['model']['num_action_chunks'] == 32
    assert actor['model']['openpi']['action_chunk'] == 32 and actor['model']['openpi']['action_horizon'] == 50
    assert cfg['rollout']['model'] == actor['model'] and cfg['rollout']['pipeline_stage_num'] == 1
    assert actor['enable_offload'] is True and cfg['rollout']['enable_offload'] is True
    assert cfg['env']['enable_offload'] is True and env['enable_offload'] is True and env['enable_init_offload'] is True
    assert [url.rstrip('/') for url in env['service_urls']] == [s['url'].rstrip('/') for s in plan['services']]
    assert 0 < row['timeout_seconds'] <= 10800
    log = Path(runner['logger']['log_path'])
    assert log.is_absolute() and log.resolve().is_relative_to((owner / row['key']).resolve())


def validate_trial_order(trials):
    assert [row['episode_steps'] for row in trials] == [32, 384], 'Require short L32 then full L384'
    assert len({row['key'] for row in trials}) == 2 and len({row['namespace'] for row in trials}) == 2


def one_arg(argv, flag):
    assert argv.count(flag) == 1, 'Service requires exactly one '+flag
    position = argv.index(flag)
    assert position + 1 < len(argv)
    return argv[position + 1]


def validate_services(services, owner):
    assert [s['physical_gpu'] for s in services] == [6, 7]
    assert len({s['key'] for s in services}) == len({s['url'].rstrip('/') for s in services}) == 2
    ports = set()
    for service in services:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', service['key'])
        argv = service['argv']
        assert isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv)
        interpreter = Path(argv[0])
        # Interpreter symlinks may target the managed Python outside this venv.
        assert interpreter.is_absolute() and interpreter.is_relative_to(ROOT)
        assert interpreter.is_file() and os.access(interpreter, os.X_OK)
        owned_path(service['cwd'])
        parsed = urllib.parse.urlparse(service['url'])
        assert parsed.scheme == 'http' and parsed.hostname == '127.0.0.1' and parsed.port
        assert parsed.path in ('', '/') and parsed.query == parsed.fragment == ''
        assert parsed.username is None and parsed.password is None
        assert parsed.port not in ports
        ports.add(parsed.port)
        assert one_arg(argv, '--physical-gpu') == str(service['physical_gpu'])
        assert one_arg(argv, '--port') == str(parsed.port)
        output = owned_path(one_arg(argv, '--output-dir'), exists=False)
        assert output.resolve().is_relative_to((owner / 'services' / service['key']).resolve())
        assert 0 < service.get('startup_seconds', 1200) <= 1800
        assert not any(k in service.get('environment', {}) for k in MASKS)


def validate(plan, frozen=False):
    assert os.getuid() == UID and socket.gethostname() == 'h100-gpu01'
    pidfd_probe()
    owner = owned_path(plan['owner_dir'], exists=frozen)
    cycle = owned_path(plan['lifecycle_path'])
    assert plan['python'] == C.load_plan(cycle)['python'], 'Use the existing RLT Python'
    repo = owned_path(plan['repo'])
    assert 'opendw' in str(repo).lower()
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == plan['repo_head']
    assert plan['physical_gpus'] == GPUS
    assert plan['source_sha256'], 'Freeze reviewed source files before launch'
    for path, digest in plan['source_sha256'].items():
        assert sha(owned_path(path)) == digest, 'Reviewed source changed: '+path
    assert plan['mode'] == 'multigpu_smoke'
    validate_trial_order(plan['trials'])
    for row in plan['trials']:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', row['key'])
        assert row['namespace'].startswith('opendw_')
        cfg = H.config(owned_path(row['config']))
        validate_config(cfg, row, plan, owner)
        owned_path(cfg['runner']['logger']['log_path'], False)
        if frozen:
            assert row['config_sha256'] == sha(row['config'])
    validate_services(plan['services'], owner)
    assert 0 < plan.get('restore_wait_seconds', 60) <= 60
    environment = read(owned_path(plan['environment_file']))
    assert not any(k in environment for k in MASKS)
    if frozen:
        assert plan['owner_script_sha256'] == sha(__file__)
        assert plan['environment_sha256'] == sha(plan['environment_file'])
        assert plan['lifecycle_plan_sha256'] == sha(cycle/'plan.json')
    return owner, cycle, repo, environment


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


def register_actors(plan, row, catalog):
    path = Path(plan['owner_dir'])/row['key']/'ray-job.json'
    selected = H.active(H.actors(plan), row['namespace'])
    if not path.exists():
        assert not selected, 'Actors exist without this driver job receipt'
        return []
    job = read(path)
    assert job['namespace'] == row['namespace']
    H.validate_actor_rows(selected, row['namespace'], {job['job_id']})
    for actor in selected:
        ident = H.proc(actor.get('pid', 0))
        if ident and ident['state'] not in ('Z', 'X'):
            catalog.add(ident, row['key'], 'exact Ray namespace/job/actor')
    return selected


def cleanup(plan, catalog, phase=None):
    trials = [r for r in plan['trials'] if phase is None or r['key'] == phase]
    selected = []
    for row in trials:
        selected.extend(register_actors(plan, row, catalog))
    catalog.scan()
    if selected:
        H.kill_actors(plan, selected)
    allowed = {row['namespace'] for row in plan['trials']}
    sent = []
    for sig, seconds in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        deadline = time.monotonic()+seconds
        while True:
            catalog.scan()
            targets = catalog.live(phase)
            if not targets:
                break
            pids = {r['pid'] for r in targets}
            for actor in H.actors(plan):
                if actor.get('state') != 'DEAD' and actor.get('pid') in pids:
                    assert actor.get('ray_namespace') in allowed, 'Target process serves unrelated actor'
                    bound = next(r for r in plan['trials'] if r['namespace'] == actor['ray_namespace'])
                    receipt = read(Path(plan['owner_dir'])/bound['key']/'ray-job.json')
                    assert actor['job_id'] == receipt['job_id']
            for ident in targets:
                catalog.send(ident, sig)
                sent.append({'pid': ident['pid'], 'start': ident['start'], 'signal': int(sig)})
            if time.monotonic() >= deadline:
                break
            time.sleep(1)
    assert not catalog.live(phase), 'Exact smoke processes did not stop'
    for row in trials:
        assert not H.active(H.actors(plan), row['namespace']), 'Smoke actors remain'
    return {'time': H.now(), 'phase': phase, 'signals': sent, 'all_stopped': True}


def http(url, endpoint='/health', post=False, timeout=5):
    request = urllib.request.Request(url.rstrip('/')+endpoint, data=b'{}' if post else None,
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def resource_snapshot(plan, catalog, phase):
    catalog.scan()
    rows = []
    for ident in catalog.live():
        path = Path('/proc')/str(ident['pid'])
        memory = {}
        try:
            for line in (path/'status').read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:')):
                    key, value = line.split(':', 1)
                    memory[key+'_kib'] = int(value.split()[0])
        except (FileNotFoundError, ProcessLookupError):
            pass
        rows.append(dict(ident, **memory))
    value = {'time': H.now(), 'phase': phase, 'processes': rows,
             'gpu_processes': H.gpu_processes(list(range(8)))}
    value['compute_memory_csv'] = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader'],
        text=True, timeout=20).strip()
    with (Path(plan['owner_dir'])/'resources.jsonl').open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    managed_pids = {r['pid'] for r in catalog.live()}
    assert not [r for r in value['gpu_processes'] if r['pid'] in managed_pids and r['gpu'] not in GPUS], \
        'Smoke created a compute/graphics context outside GPU4-7'
    for service in plan['services']:
        service_pids = {r['pid'] for r in catalog.live('service_'+service['key'])}
        assert not [r for r in value['gpu_processes'] if r['pid'] in service_pids
                    and r['gpu'] != service['physical_gpu']], 'WM service used a different physical GPU'
    return value


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


def run_driver(plan, key):
    owner, cycle, repo, env = validate(plan, True)
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
            assert self.get_world_size(name) == 2
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
    owner, cycle, repo, base_env = validate(input_plan)
    owner.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (owner/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (owner/'owner-identity.json').exists(), 'Do not replay an owner'
    assert not (cycle/'rlt-stopped.json').exists(), 'Owner must witness its own borrowing'
    plan = copy.deepcopy(input_plan)
    cp = C.load_plan(cycle)
    plan.update(token=uuid.uuid4().hex, ray_address=cp['ray_address'], ray_dashboard_url=cp['ray_dashboard_url'],
                management_namespace='opendw_ops_'+owner.name[-40:], uid=UID,
                owner_script_sha256=sha(__file__), environment_sha256=sha(plan['environment_file']),
                lifecycle_plan_sha256=sha(cycle/'plan.json'))
    for row in plan['trials']:
        row['config_sha256'] = sha(row['config'])
        assert not H.active(H.actors(cp), row['namespace'])
        (owner/row['key']).mkdir(exist_ok=False)
    for service in plan['services']:
        (owner/'services'/service['key']).mkdir(parents=True, exist_ok=False)
    record(owner/'owner-plan.json', plan)
    record(owner/'owner-identity.json', dict(H.proc(os.getpid()), time=H.now()))
    catalog = Catalog(owner/'process-catalog.json', plan['token'])
    closing = {'value': False, 'launching': False, 'signal': None}
    def terminate(sig, frame):
        closing['signal'] = sig
        if not closing['value'] and not closing['launching']:
            raise RuntimeError('Owner received signal '+str(sig))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    children = []
    def launch(argv, cwd, env, phase, log):
        closing['launching'] = True
        try:
            env = dict(env, **{TOKEN: plan['token'], PHASE: phase})
            with Path(log).open('x') as stream:
                child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            ident = H.proc(child.pid)
            catalog.add(ident, phase, 'exact Popen child')
            children.append(child)
        finally:
            closing['launching'] = False
        if closing['signal']:
            raise RuntimeError('Termination requested during launch')
        return child
    error = None
    borrowed = False
    results = []
    terminal = 'failed'
    try:
        add_allowlist(plan)
        # Each service initializes on CPU. Both health/PID proofs are complete
        # before the separately reviewed four-card borrow transaction begins.
        service_children = {}
        for service in plan['services']:
            port = urllib.parse.urlparse(service['url']).port
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
            env = dict(base_env)
            env.update(service.get('environment', {}))
            env['CUDA_VISIBLE_DEVICES'] = str(service['physical_gpu'])
            for key in MASKS[1:]:
                env.pop(key, None)
            target = owner/'services'/service['key']
            service_child = launch(service['argv'], service['cwd'], env, 'service_'+service['key'], target/'service.log')
            service_children[service['key']] = service_child
            deadline = time.monotonic()+service.get('startup_seconds', 1200)
            while True:
                assert all(p.poll() is None for p in service_children.values()), 'Service exited before CPU readiness'
                resource_snapshot(plan, catalog, 'service_cpu_load_'+service['key'])
                live_pids = {r['pid'] for r in catalog.live()}
                assert not [r for r in H.gpu_processes(list(range(8))) if r['pid'] in live_pids], 'Service used GPU before borrowing'
                try:
                    health = http(service['url'])
                except (OSError, ValueError):
                    health = None
                if health:
                    assert health['ok'] and health['is_offloaded'] and health['pid'] == service_child.pid
                    assert health['physical_gpu'] == service['physical_gpu']
                    record(target/'service-cpu-ready.json', health)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Service CPU startup deadline: '+service['key'])
                time.sleep(5)
        # Complete the exact stop/receipt transaction before reacting to a
        # termination; finally must have a conclusive borrowing receipt.
        closing['launching'] = True
        try:
            with (cycle/'operation.lock').open('a') as operation_lock:
                fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                C.stop(cycle)
                assert (cycle/'rlt-stopped.json').is_file(), 'Four-card stop returned without a complete receipt'
                borrowed = True
        finally:
            closing['launching'] = False
        if closing['signal']:
            raise RuntimeError('Termination requested during RLT borrowing')
        for row in plan['trials']:
            env = dict(base_env)
            for key in MASKS:
                env.pop(key, None)
            env.update(PYTHONPATH=str(repo)+':'+env.get('PYTHONPATH', ''), RAY_ADDRESS=plan['ray_address'])
            argv = [plan['python'], '-u', '-B', str(Path(__file__).resolve()), '--plan', str(owner/'owner-plan.json'),
                    'driver', '--key', row['key']]
            child = launch(argv, str(repo), env, row['key'], owner/row['key']/'driver.log')
            started = time.monotonic()
            while child.poll() is None:
                register_actors(plan, row, catalog)
                resource_snapshot(plan, catalog, row['key'])
                assert all(p.poll() is None for p in service_children.values()), 'Service exited during smoke'
                if time.monotonic()-started >= row['timeout_seconds']:
                    terminal = 'timed_out'
                    raise TimeoutError('Smoke deadline: '+row['key'])
                atomic(owner/'state.json', {'time': H.now(), 'phase': row['key'], 'elapsed_seconds': time.monotonic()-started})
                time.sleep(10)
            cleanup_result = cleanup(plan, catalog, row['key'])
            record(owner/row['key']/'cleanup.json', cleanup_result)
            result = {'key': row['key'], 'num_envs': row['num_envs'], 'exit_code': child.returncode,
                      'seconds': time.monotonic()-started}
            results.append(result)
            record(owner/row['key']/'result.json', result)
            assert child.returncode == 0, 'Smoke failed; do not start next N: '+row['key']
            assert read(owner/row['key']/'driver-finished.json')['exit_code'] == 0
            assert (owner/row['key']/'verified-placement.json').is_file()
            for service in plan['services']:
                response = http(service['url'], '/offload', post=True, timeout=120)
                assert response['ok'] and response['is_offloaded']
                health = http(service['url'])
                assert health['ok'] and health['is_offloaded']
                assert health['pid'] == service_children[service['key']].pid
                assert health['physical_gpu'] == service['physical_gpu']
            resource_snapshot(plan, catalog, row['key']+'_complete')
        terminal = 'completed'
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
        record(owner/'error.json', error)
    finally:
        closing['value'] = True
        recovery_error = None
        try:
            cleanup_result = cleanup(plan, catalog)
            record(owner/'cleanup.json', cleanup_result)
            assert cleanup_result['all_stopped'] and not catalog.live(), 'Owner cleanup proof is incomplete'
            for child in children:
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            full_stop_receipt = (cycle/'rlt-stopped.json').is_file()
            if not full_stop_receipt and (cycle/'clean-old-stop-attempt.json').exists():
                # Only our workers are proven gone. Original RLT may still be
                # healthy on some cards, so this is not an all-GPUs-empty claim.
                borrowed = False
                partial = {'time': H.now(), 'cycle_id': cycle.name, 'gpus': GPUS,
                           'terminal_status': terminal, 'all_workers_stopped': True,
                           'managed_processes': list(catalog.rows.values())}
                receipt = owner/'partial-smoke-release.json'
                record(receipt, partial)
                with (cycle/'operation.lock').open('a') as operation_lock:
                    fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    recovered = C.recover_partial(cycle, receipt)
                record(owner/'partial-rlt-recovery.json', recovered)
            elif not full_stop_receipt and (cycle/'clean-old-stopped.json').exists():
                raise RuntimeError('Partial RLT completion marker has no top-level stop-attempt receipt')
            else:
                borrowed = borrowed or full_stop_receipt
            if borrowed:
                assert full_stop_receipt, 'A full return requires the complete four-card stopped receipt'
                assert not H.gpu_processes(GPUS), 'GPU4-7 compute/graphics not released'
                release = {'time': H.now(), 'cycle_id': cycle.name, 'gpus': GPUS, 'terminal_status': terminal,
                           'all_workers_stopped': True, 'managed_processes': list(catalog.rows.values())}
                receipt = owner/'smoke-release.json'
                record(receipt, release)
                with (cycle/'operation.lock').open('a') as operation_lock:
                    fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    dispatched = H.resume(cycle, receipt)
                record(owner/'rlt-return-dispatched.json', {'time': H.now(), 'result': dispatched})
                deadline = time.monotonic()+plan.get('restore_wait_seconds', 60)
                while True:
                    state = H.status(cycle)
                    atomic(owner/'rlt-status.json', state)
                    assert set(state['runs']) == {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
                    if state['all_first_rounds_verified']:
                        record(owner/'rlt-first-round.json', state)
                        break
                    if any(run.get('finished') for run in state['runs'].values()):
                        raise RuntimeError('Restored RLT exited before first-round validation')
                    if time.monotonic() >= deadline:
                        record(owner/'rlt-first-round-pending.json', {'time': H.now(),
                            'status': 'return_dispatched_first_round_pending', 'state': state})
                        break
                    time.sleep(15)
        except BaseException as exc:
            recovery_error = {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
            record(owner/'recovery-error.json', recovery_error)
        record(owner/'final.json', {'time': H.now(), 'mode': plan['mode'], 'physical_gpus': GPUS,
            'terminal_status': terminal, 'trials': results,
            'error': error, 'recovery_error': recovery_error, 'rlt_borrowed': borrowed,
            'partial_rlt_recovery_recorded': (owner/'partial-rlt-recovery.json').exists(),
            'rlt_return_dispatched': (owner/'rlt-return-dispatched.json').exists(),
            'rlt_first_round_pending': (owner/'rlt-first-round-pending.json').exists(),
            'rlt_first_round_verified': (owner/'rlt-first-round.json').exists(),
            'gpu_processes': H.gpu_processes(list(range(8)))})
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
