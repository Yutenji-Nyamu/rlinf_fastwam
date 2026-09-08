import ast

import datetime

import hashlib

import json

import math

import os

import pwd

import runpy

import shlex

import shutil

import signal

import subprocess

import sys

import time

from pathlib import Path

def now():
    return datetime.datetime.now().astimezone().isoformat()

def save(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)

def attempt(phase):
    save(ST / (phase + '-attempt.json'), {'time': now()}, exclusive=True)

def cmd(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()

def git(root, *args):
    return cmd('git', '-C', str(root), *args)

def flat(value, prefix=''):
    if not isinstance(value, dict):
        return {prefix: value}
    return {key: child for name, item in value.items()
            for key, child in flat(item, (prefix + '.' if prefix else '') + name).items()}

def diff(before, after):
    before, after = flat(before), flat(after)
    return {key: [before.get(key), after.get(key)] for key in sorted(before.keys() | after.keys())
            if before.get(key) != after.get(key)}

def proc(pid):
    path = Path('/proc') / str(pid)
    return {'pid': int(pid), 'uid': path.stat().st_uid,
            'start': int((path / 'stat').read_text().rsplit(')', 1)[1].split()[19]),
            'cmd': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}

def gpu_snapshot():
    devices = {}
    for line in cmd('nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits').splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(',')]
        devices[int(index)] = {'uuid': uuid, 'memory_mib': int(memory),
                               'utilization': int(utilization), 'processes': []}
    by_uuid = {value['uuid']: value for value in devices.values()}
    for line in cmd('nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                    '--format=csv,noheader,nounits').splitlines():
        if not line.strip():
            continue
        uuid, pid = [part.strip() for part in line.split(',')]
        by_uuid[uuid]['processes'].append(proc(int(pid)))
    return devices

def assert_processes_unchanged(protected):
    for pid, expected in protected.items():
        actual = proc(pid)
        # Ray changes its process title as methods run; identity is PID/UID/start.
        assert all(actual[key] == expected[key] for key in ('pid', 'uid', 'start')), {
            'expected': expected, 'actual': actual}

def source_manifest():
    manifest = json.loads((ST / 'source-manifest.json').read_text())
    entries = manifest['files'] if isinstance(manifest, dict) else manifest
    result = []
    for entry in entries:
        relative = entry['path'] if isinstance(entry, dict) else entry
        path = ROOT / relative
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert path.is_file() and not path.is_symlink() and path.stat().st_size < 1000000
        if isinstance(entry, dict) and 'sha256' in entry:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], relative
        result.append(relative)
    assert result and len(result) == len(set(result))
    return result

def cleanup_namespace(namespace, receipt_path):
    """Kill only this driver's verified owned actors, never another Ray job."""
    import ray
    from ray.util.state import list_actors
    if not ray.is_initialized():
        return
    assert namespace in {NAMESPACE, NAMESPACE + '_validate'}
    job_id = ray.get_runtime_context().get_job_id()
    job_id = job_id.hex() if hasattr(job_id, 'hex') else str(job_id)
    # GCS names and the State API are updated asynchronously (notably NodeProbe
    # can disappear from names before its ALIVE state catches up). Wait for a
    # consistent identity view before authorizing any actor termination.
    deadline = time.monotonic() + 10.0
    for _ in range(20):
        before = {(row['namespace'], row['name']) for row in ray.util.list_named_actors(all_namespaces=True)}
        rows = [row.asdict() if hasattr(row, 'asdict') else dict(row) for row in
                list_actors(detail=True, filters=[('state', '=', 'ALIVE')], limit=2000, timeout=3)]
        selected = [row for row in rows if row.get('ray_namespace') == namespace]
        targets = {(namespace, row['name']) for row in selected if row.get('name')}
        named = {item for item in before if item[0] == namespace}
        if named == targets:
            assert all(row['job_id'] == job_id for row in selected), selected
            try:
                identities = [proc(row['pid']) for row in selected]
            except FileNotFoundError:
                identities = None
            if identities is not None:
                assert all(item['uid'] == 1003 for item in identities)
                break
        if time.monotonic() >= deadline:
            raise RuntimeError({'message': 'Actor identity views did not converge; nothing killed',
                                'namespace': namespace, 'named': sorted(named), 'states': selected})
        time.sleep(0.5)
    else:
        raise RuntimeError('Actor identity views did not converge; nothing killed')
    save(receipt_path, {'time': now(), 'namespace': namespace, 'job_id': job_id,
                        'processes': identities, 'targets': sorted(targets), 'status': 'verified'}, exclusive=True)
    managers = {'CollectiveManager', 'DeviceLockManager', 'NodeManager', 'PortLockManager', 'WorkerManager'}
    for _, name in sorted(targets, key=lambda item: (item[1] in managers, item[1])):
        try:
            ray.kill(ray.get_actor(name, namespace=namespace), no_restart=True)
        except ValueError:
            pass
    after = {(row['namespace'], row['name']) for row in ray.util.list_named_actors(all_namespaces=True)}
    # Names outside our namespace must not be removed by this operation.
    missing = (before - targets) - after
    save(receipt_path, {'time': now(), 'namespace': namespace, 'job_id': job_id,
                        'processes': identities, 'targets': sorted(targets),
                        'status': 'completed', 'unrelated_missing_names': sorted(missing)})
    assert not missing, missing

def commit():
    checked = json.loads((ST / 'tests-receipt.json').read_text())
    assert checked['passed'] and (PRE / 'contract.json').exists()
    files = source_manifest()
    for relative, expected in checked['source_hashes'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert git(ROOT, 'rev-parse', 'HEAD') == BASE_HEAD and git(ROOT, 'branch', '--show-current') == BRANCH
    assert not git(ROOT, 'diff', '--cached', '--name-only')
    for source in sorted((ST / 'tests').glob('test_*.py')):
        destination = ROOT / 'tests/unit_tests' / source.name
        if destination.exists():
            assert destination.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, destination)
        relative = str(destination.relative_to(ROOT))
        if relative not in files:
            files.append(relative)
    git(ROOT, 'add', '--', *files)
    assert set(git(ROOT, 'diff', '--cached', '--name-only').splitlines()) == set(files)
    git(ROOT, 'diff', '--cached', '--check')
    (PRE / 'reviewed.patch').write_text(git(ROOT, 'diff', '--cached'))
    attempt('commit')
    git(ROOT, 'commit', '-m', 'Add independent online BC batch two-level DVAC weighting')
    assert not git(ROOT, 'status', '--porcelain')
    save(ST / 'source-receipt.json', {'time': now(), 'base': BASE_HEAD, 'head': git(ROOT, 'rev-parse', 'HEAD'),
                                     'branch': BRANCH, 'files': files})
    print((ST / 'source-receipt.json').read_text(), flush=True)

def launch():
    source = json.loads((ST / 'source-receipt.json').read_text())
    assert git(ROOT, 'rev-parse', 'HEAD') == source['head'] and not git(ROOT, 'status', '--porcelain')
    assert git(ROOT, 'branch', '--show-current') == BRANCH
    assert not RUN.exists()
    devices = gpu_snapshot()
    assert_empty(devices)
    protected = protected_processes(devices)
    assert protected
    assert shutil.disk_usage('/data/chenyiteng').free > 20 * 1024 ** 3
    available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')).split()[1])
    assert available > 200 * 1024 ** 2
    # Fail rather than reuse any existing namespace, including an old smoke.
    import ray
    ray.init(address=RAY_ADDRESS, namespace=NAMESPACE + '_prelaunch', logging_level='ERROR')
    try:
        names = ray.util.list_named_actors(all_namespaces=True)
        assert not any(row['namespace'] == NAMESPACE for row in names)
    finally:
        ray.shutdown()
    assert_empty(gpu_snapshot())
    assert_processes_unchanged(protected)
    attempt('launch')
    RUN.mkdir(parents=True)
    runtime = RUN / 'runtime'
    shutil.copytree(PRE, runtime)
    (runtime / 'source-head.txt').write_text(source['head'] + '\n')
    save(runtime / 'protected-before.json', {str(key): value for key, value in protected.items()})
    save(runtime / 'gpu-before.json', devices)
    env = os.environ.copy()
    for key in ('CUDA_VISIBLE_DEVICES', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env.update(json.loads((runtime / 'environment.json').read_text()))
    with (runtime / 'wrapper.log').open('xb') as stream:
        process = subprocess.Popen(['bash', str(runtime / 'wrapper.sh'), str(runtime)], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = proc(process.pid)
    (runtime / 'wrapper.pid').write_text(str(process.pid) + '\n')
    save(ST / 'launch-receipt.json', {'time': now(), 'root': str(ROOT), 'run': str(RUN),
        'wrapper': identity, 'source_head': source['head'], 'namespace': NAMESPACE,
        'protected_processes': list(protected.values()), 'physical_gpus': list(GPUS)})
    assert_processes_unchanged(protected)
    save(runtime / 'protected-after-launch.json', {'time': now(), 'unchanged': True,
                                                  'processes': list(protected.values())})
    print((ST / 'launch-receipt.json').read_text(), flush=True)

def smoke_driver():
    sys.path.insert(0, str(ROOT))
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE = NAMESPACE
    runtime = RUN / 'runtime'
    save(runtime / 'driver-identity.json', {**proc(os.getpid()), 'namespace': NAMESPACE}, exclusive=True)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    sys.argv = [str(ROOT / 'examples/embodiment/train_embodied_agent.py'), *sys.argv[2:]]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        import ray
        try:
            # Our completed workers may notify their driver when explicitly
            # released; suppress that failure hook only during owned teardown.
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            assert Cluster.NAMESPACE == NAMESPACE
            cleanup_namespace(NAMESPACE, runtime / 'owned-cleanup.json')
        finally:
            ray.shutdown()

ST = Path('/data/chenyiteng/results/server-maintenance-20260908/bc-dvac-new')
BASE = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-dvac-w5')
ROOT = BASE.parent / 'pi05-online-bc-dvac-new'
BASE_HEAD = 'ad3da329270960f4ad378842e7138bbde29c521e'
BRANCH = 'codex/sz-pi05-online-bc-dvac-new-20260908'
RUN = Path('/home/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-bc-dvac-new-4u5-smoke2-gpu7-20260908-v1')
OLD_RUN = Path('/home/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc-dvac4x1-b1024-u5-m10-seed42-w0to5-eval8x4-gpu7-formal100-20260908-v1')
PRE = ST / 'prepared'
PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
NAMESPACE = 'RLinf_bc_dvac_new_smoke7_20260908'
RAY_ADDRESS = '172.17.0.1:6389'
GPUS = (7,)

def assert_empty(devices):
    state = devices[7]
    assert not state['processes'] and state['memory_mib'] < 128 and state['utilization'] == 0, state


def protected_processes(devices):
    return {entry['pid']: entry for gpu, value in devices.items() if gpu != 7 for entry in value['processes']}


def tests():
    PRE.mkdir(exist_ok=True)
    files = source_manifest()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', RLINF_BC_NEW_SOURCE_ROOT=str(ROOT))
    candidates = list((ST / 'tests').glob('test_*.py'))
    candidates.extend(sorted((ROOT / 'tests').rglob('test_online_bc*.py')))
    assert candidates
    for relative in files:
        if relative.endswith('.py'):
            ast.parse((ROOT / relative).read_text(), filename=relative)
    result = subprocess.run([PY, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *map(str, candidates)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (PRE / 'tests.txt').write_text(result.stdout)
    print(result.stdout, flush=True)
    assert result.returncode == 0
    git(ROOT, 'diff', '--check')
    save(ST / 'tests-receipt.json', {'time': now(), 'passed': True, 'files': list(map(str, candidates)), 'source_hashes': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files}})


def prepare():
    import yaml
    import ray
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    PRE.mkdir(exist_ok=True)
    oldcfg = yaml.safe_load((OLD_RUN / 'runtime/resolved.yaml').read_text())
    env = dict(PYTHONPATH=str(ROOT) + ':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
               REPO_PATH=str(ROOT), EMBODIED_PATH=str(ROOT / 'examples/embodiment'),
               ASSETS_PATH=oldcfg['env']['train']['assets_path'], PI05_MODEL_PATH=oldcfg['actor']['model']['model_path'],
               ONLINE_BC_RUN_DIR=str(RUN), RAY_ADDRESS=RAY_ADDRESS, RLINF_CODE_WORKING_DIR=str(ROOT),
               PYTHONDONTWRITEBYTECODE='1', TORCHINDUCTOR_COMPILE_THREADS='1')
    os.environ.update(env)
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    sys.path.insert(0, str(ROOT))
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.runner_utils import check_progress
    for step in (0, 1, 2):
        result = check_progress(step, 2, -1, -1, 1.0, run_time_exceeded=False)
        assert not result[0] and not result[1]
    config_name = 'robotwin_adjust_bottle_online_bc_openpi'
    overrides = ['+online_bc_model=pi05_sidney', '+bc_dvac=two_level_batch', '+rollout.seed=42',
                 'env.train.total_num_envs=4', 'algorithm.update_epoch=5', 'runner.max_epochs=2',
                 'runner.val_check_interval=-1', 'runner.save_interval=-1',
                 'actor.optim.total_training_steps=1000',
                 'runner.logger.experiment_name=pi05-bc-dvac-new-4u5-smoke2-gpu7-0908-v1',
                 r'cluster.component_placement={actor\,env\,rollout:"7"}',
                 'algorithm.online_bc.dvac.debug_batches=2']
    Cluster.NAMESPACE = NAMESPACE + '_validate'
    with initialize_config_dir(version_base='1.1', config_dir=str(ROOT / 'examples/embodiment/config')):
        cfg = compose(config_name=config_name, overrides=overrides)
        try:
            cfg = validate_cfg(cfg)
        finally:
            try:
                cleanup_namespace(Cluster.NAMESPACE, PRE / 'validation-cleanup.json')
            finally:
                ray.shutdown()
    actual = OmegaConf.to_container(cfg, resolve=True)
    changes = diff(oldcfg, actual)
    identity = {'runner.logger.experiment_name', 'runner.logger.log_path', 'runner.per_worker_log_path',
                'env.train.task_config.save_path', 'env.eval.task_config.save_path',
                'env.train.video_cfg.video_base_dir', 'env.eval.video_cfg.video_base_dir',
                'env.train.seeds_path', 'env.eval.seeds_path', 'algorithm.online_bc.data_path'}
    smoke = {'runner.max_epochs', 'runner.val_check_interval', 'runner.save_interval'}
    unexpected = {key: value for key, value in changes.items()
                  if key not in identity | smoke and not key.startswith('algorithm.online_bc.dvac.')}
    assert not unexpected, unexpected
    assert cfg.actor.global_batch_size == 1024 and cfg.actor.micro_batch_size == 32
    assert cfg.algorithm.update_epoch == 5 and cfg.env.train.total_num_envs == 4
    assert cfg.actor.model.num_steps == 10 and cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.optim.lr == oldcfg['actor']['optim']['lr']
    assert cfg.algorithm.online_bc.max_success_chunks is None
    assert list(actual['cluster']['component_placement'].values()) == ['7']
    assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None and not RUN.exists()
    for split in ('train', 'eval'):
        assert hashlib.sha256(Path(oldcfg['env'][split]['seeds_path']).read_bytes()).digest() == hashlib.sha256(Path(actual['env'][split]['seeds_path']).read_bytes()).digest()
    OmegaConf.save(cfg, PRE / 'resolved.yaml', resolve=True)
    argv = [PY, '-B', str(ST / 'ops.py'), 'smoke_driver', '--config-name', config_name, *overrides]
    save(PRE / 'argv.json', argv)
    save(PRE / 'environment.json', env)
    (PRE / 'command.txt').write_text(shlex.join(argv) + '\n')
    (PRE / 'command.sh').write_text('#!/usr/bin/env bash\nexec ' + shlex.join(argv) + '\n')
    (PRE / 'wrapper.sh').write_text('''#!/usr/bin/env bash
set +e
runtime=$1
ulimit -n 4096 || exit 91
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 3600s bash "$runtime/command.sh" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
''')
    for name in ('wrapper.sh', 'command.sh'):
        subprocess.run(['bash', '-n', str(PRE / name)], check=True)
    save(PRE / 'config_diff.json', changes)
    save(PRE / 'contract.json', {'time': now(), 'root': str(ROOT), 'run': str(RUN), 'base': BASE_HEAD,
         'physical_gpus': [7], 'namespace': NAMESPACE, 'rounds': 2, 'attempts_per_round': 4,
         'adam_per_round': 5, 'global_batch': 1024, 'micro_batch': 32, 'denoise_steps': 10,
         'learning_rate': cfg.actor.optim.lr, 'alpha_local': 1.0, 'alpha_chunk': 1.0,
         'save_interval': -1, 'eval_interval': -1, 'timeout_seconds': 3600,
         'length_filter': None, 'formal_replacement': False})
    print(json.dumps({'config_diff': changes, 'contract': json.loads((PRE / 'contract.json').read_text())}), flush=True)


def status():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    runtime = RUN / 'runtime'
    out = {'time': now(), 'run': str(RUN), 'gpu': gpu_snapshot()[7]}
    for name in ('exit_code.txt', 'driver-identity.json', 'owned-cleanup.json'):
        if (runtime / name).is_file():
            out[name] = (runtime / name).read_text()
    if (runtime / 'driver.log').is_file():
        with (runtime / 'driver.log').open('rb') as stream:
            stream.seek(max(0, (runtime / 'driver.log').stat().st_size - 7000))
            out['log_tail'] = stream.read().decode(errors='replace')
    if (RUN / 'tensorboard').is_dir():
        ea = EventAccumulator(str(RUN / 'tensorboard'), size_guidance={'scalars': 0})
        ea.Reload()
        out['scalars'] = {tag: [{'step': x.step, 'value': x.value} for x in ea.Scalars(tag)] for tag in ea.Tags()['scalars']}
    save(ST / 'status.json', out)
    print(json.dumps(out), flush=True)


def verify():
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    sys.path.insert(0, str(ROOT))
    from rlinf.algorithms.online_bc_dvac_two_level import compute_two_level_bc_weights
    runtime = RUN / 'runtime'
    assert (runtime / 'exit_code.txt').read_text().strip() == '0'
    assert json.loads((runtime / 'owned-cleanup.json').read_text())['status'] == 'completed'
    ea = EventAccumulator(str(RUN / 'tensorboard'), size_guidance={'scalars': 0}); ea.Reload()
    series = {tag: [{'step': x.step, 'value': x.value} for x in ea.Scalars(tag)] for tag in ea.Tags()['scalars']}
    assert all(math.isfinite(x['value']) for values in series.values() for x in values)
    def points(suffix):
        keys = [tag for tag in series if tag == suffix or tag.endswith('/' + suffix)]
        assert len(keys) == 1, (suffix, keys)
        return series[keys[0]]
    assert len(points('env/success_once')) == 2
    assert len(points('bc/actor_loss')) >= 1
    for value in points('dvac_new/weight_mean'):
        assert abs(value['value'] - 1) < 2e-6
    assert all(x['value'] == 1024 for x in points('dvac_new/total_query_count'))
    assert any(x['value'] > 0 for x in points('dvac_new/chunk_std'))
    assert any(x['value'] > 0 for x in points('actor/grad_norm'))
    debug = sorted((RUN / 'dvac_new_debug').glob('update_*.pt'))
    assert len(debug) == 2
    checks = []
    for path in debug:
        tensor = torch.load(path, weights_only=True)
        weight, metrics = compute_two_level_bc_weights(tensor['variance'], tensor['mask'], **tensor['settings'])
        torch.testing.assert_close(weight, tensor['weights'], rtol=0, atol=0)
        checks.append({'path': str(path), 'shape': list(weight.shape), 'exact_recompute': True, 'metrics': metrics})
    protected = json.loads((runtime / 'protected-before.json').read_text())
    assert_processes_unchanged({int(key): value for key, value in protected.items()})
    assert_empty(gpu_snapshot())
    assert not list(RUN.glob('**/checkpoints/global_step_*'))
    out = {'time': now(), 'passed': True, 'source_head': (runtime / 'source-head.txt').read_text().strip(),
           'series': series, 'debug': checks, 'protected_unchanged': True, 'gpu7_released': True}
    save(ST / 'smoke-verification.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    assert os.getuid() == 1003 and pwd.getpwuid(os.getuid()).pw_name == 'chenyiteng'
    {'prepare': prepare, 'tests': tests, 'commit': commit, 'launch': launch,
     'smoke_driver': smoke_driver, 'verify': verify, 'status': status}[sys.argv[1]]()
