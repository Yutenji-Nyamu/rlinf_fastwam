"""Prepare/launch only authorized GPU6 AttenA-FK smoke and 4/U5 formal jobs.

Inputs, written by the audited source/calibration workflow:
  source-receipt.json: {head, base, branch}
  tests-receipt.json: {passed, source_hashes: {relative_path: sha256}}
  attena-launch-settings.json: {attena: {...}, assets: [{path, sha256}, ...]}
  smoke-verification.json: {passed: true, checkpoint_restore_passed: true}

Usage: launch_ops.py prepare|launch|driver|status smoke|formal
No code publication, GPU polling loop, unrelated process stop, or Ray restart.
"""
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
import runpy
import shlex
import shutil
import signal
import subprocess
import sys

ST = Path('/data/chenyiteng/results/server-maintenance-20260909/bc-attena-fk')
ROOT = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-attena-fk')
BRANCH = 'codex/sz-pi05-online-bc-attena-fk-20260909'
BASE = '01d770db3988da7862454e97434d4ff08f726fa2'
PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
RAY_ADDRESS = '172.17.0.1:6389'
RESULTS = Path('/home/chenyiteng/results/rlinf-shenzhen/online-bc')
BASELINE = RESULTS / 'pi05-pillbottle-bc4x1-b1024-u5-m10-seed42-eval8x4-gpu6-formal100-20260908-v1'
GPU = 6
SPECS = {
    'smoke': {'rounds': 2, 'eval': -1, 'save': 2, 'debug': 2, 'timeout': 3600},
    'formal': {'rounds': 100, 'eval': 5, 'save': 10, 'debug': 0, 'timeout': 172800},
}
HELPER = Path('/data/chenyiteng/results/server-maintenance-20260908/bc-dvac-new/ops.py')
spec = importlib.util.spec_from_file_location('verified_bc_namespace_helpers', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def now():
    return datetime.datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(path.read_text())


def save(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(key):
    cfg = SPECS[key]
    suffix = f'{key}{cfg["rounds"]}'
    run = RESULTS / f'pi05-bc-attena-fk-sum-4u5-len3-gpu6-{suffix}-20260909-v1'
    namespace = f'RLinf_bc_attena_fk_sum_4u5_gpu6_{key}_20260909'
    pre = ST / ('prepared-' + key)
    return cfg, run, namespace, pre


def identity_check():
    assert os.getuid() == 1003 and pwd.getpwuid(os.getuid()).pw_name == 'chenyiteng'
    assert subprocess.check_output(['hostname'], text=True).strip() == 'admin'
    assert ROOT.parent == Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')


def source_check():
    source = read(ST / 'source-receipt.json')
    tests = read(ST / 'tests-receipt.json')
    assert source['base'] == BASE and source['branch'] == BRANCH
    assert h.git(ROOT, 'rev-parse', 'HEAD') == source['head']
    assert h.git(ROOT, 'branch', '--show-current') == BRANCH
    assert not h.git(ROOT, 'status', '--porcelain'), 'Dedicated source tree is not clean'
    assert tests['passed'] and tests['source_hashes']
    actual = {}
    for relative, expected in tests['source_hashes'].items():
        path = ROOT / relative
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file()
        actual[relative] = sha(path)
        assert actual[relative] == expected, relative
    return source['head'], actual


def settings_check():
    settings = read(ST / 'attena-launch-settings.json')
    attena = settings['attena']
    allowed = {'enabled', 'geometry_path', 'calibration_path', 'ell', 'epsilon', 'clip_max', 'debug_batches'}
    assert set(attena) <= allowed and (allowed - {'debug_batches'}) <= set(attena), attena
    assert attena['enabled'] is True
    assert attena['ell'] == 0.1 and attena['epsilon'] == 0.001 and attena['clip_max'] == 2.0
    declared = {str(Path(row['path'])): row['sha256'] for row in settings['assets']}
    assert len(declared) == len(settings['assets'])
    expected_paths = {str(Path(attena[key])) for key in ('geometry_path', 'calibration_path')}
    assert expected_paths <= set(declared), 'Both geometry and calibration must be content-locked'
    assets = {}
    for name, expected in declared.items():
        path = Path(name)
        assert path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
        assert path.is_file() and not path.is_symlink()
        assets[name] = sha(path)
        assert assets[name] == expected, name
    return settings, assets


def clean_env():
    for key in ('CUDA_VISIBLE_DEVICES', 'LD_PRELOAD', 'RLINF_SCENE_FENCE_LIBRARY',
                'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)


def environment(run, baseline):
    return {
        'PYTHONPATH': str(ROOT) + ':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
        'REPO_PATH': str(ROOT), 'EMBODIED_PATH': str(ROOT / 'examples/embodiment'),
        'ASSETS_PATH': baseline['env']['train']['assets_path'],
        'PI05_MODEL_PATH': baseline['actor']['model']['model_path'],
        'ONLINE_BC_RUN_DIR': str(run), 'RAY_ADDRESS': RAY_ADDRESS,
        'RLINF_CODE_WORKING_DIR': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
        'TORCHINDUCTOR_COMPILE_THREADS': '1',
    }


def cleanup(namespace, receipt):
    h.NAMESPACE = namespace.removesuffix('_validate')
    h.cleanup_namespace(namespace, receipt)


def empty_gpu(devices):
    gpu = devices[GPU]
    assert not gpu['processes'] and gpu['memory_mib'] < 128 and gpu['utilization'] == 0, gpu


def protected_processes(devices):
    return {item['pid']: item for gpu, device in devices.items() if gpu != GPU for item in device['processes']}


def check_budget(actual, settings, key):
    cfg = SPECS[key]
    assert actual['env']['train']['total_num_envs'] == 4
    assert actual['algorithm']['update_epoch'] == 5
    assert actual['actor']['global_batch_size'] == 1024 and actual['actor']['micro_batch_size'] == 32
    assert actual['actor']['model']['num_steps'] == 10 and actual['actor']['model']['num_action_chunks'] == 50
    assert actual['actor']['model']['action_dim'] == 14
    assert actual['actor']['optim']['lr'] == 2.5e-5 and actual['actor']['optim']['lr_scheduler'] == 'constant'
    assert actual['algorithm']['online_bc']['max_success_chunks'] == 3
    assert actual['algorithm']['online_bc']['demo_weight'] == 0
    assert not actual['algorithm']['online_bc'].get('dvac', {}).get('enabled', False)
    expected_method = copy.deepcopy(settings['attena'])
    expected_method['debug_batches'] = cfg['debug']
    assert actual['algorithm']['online_bc']['attena'] == expected_method
    assert actual['runner']['max_epochs'] == cfg['rounds']
    assert actual['runner']['val_check_interval'] == cfg['eval'] and actual['runner']['save_interval'] == cfg['save']
    assert actual['runner']['resume_dir'] is None and actual['runner']['ckpt_path'] is None
    assert list(actual['cluster']['component_placement'].values()) == [str(GPU)]
    assert actual['env']['eval']['total_num_envs'] == 8
    assert actual['env']['eval']['rollout_epoch'] == 4 and actual['env']['eval']['fixed_reset_batch_count'] == 4
    assert actual['env']['eval']['use_fixed_reset_state_ids'] is True


def prepare(key):
    import ray
    import yaml
    from omegaconf import OmegaConf
    cfg_spec, run, namespace, pre = select(key)
    head, hashes = source_check()
    settings, assets = settings_check()
    assert not run.exists(), 'Run identity already used'
    pre.mkdir(exist_ok=True)
    assert not (pre / 'contract.json').exists() and not (pre / 'validation-cleanup.json').exists()
    baseline_path = BASELINE / 'runtime/resolved.yaml'
    baseline = yaml.safe_load(baseline_path.read_text())
    assert baseline['runner']['max_epochs'] == 100 and baseline['algorithm']['update_epoch'] == 5
    assert baseline['env']['train']['total_num_envs'] == 4
    actual = copy.deepcopy(baseline)
    # Copy the complete, actually used clean baseline. Only explicit identity,
    # admission/method, and bounded smoke fields are changed below.
    actual['runner']['logger']['log_path'] = str(run)
    actual['runner']['logger']['experiment_name'] = run.name
    actual['algorithm']['online_bc']['data_path'] = str(run / 'success_data')
    actual['algorithm']['online_bc']['max_success_chunks'] = 3
    actual['algorithm']['online_bc']['attena'] = copy.deepcopy(settings['attena'])
    actual['algorithm']['online_bc']['attena']['debug_batches'] = cfg_spec['debug']
    actual['runner']['max_epochs'] = cfg_spec['rounds']
    actual['runner']['val_check_interval'] = cfg_spec['eval']
    actual['runner']['save_interval'] = cfg_spec['save']
    actual['cluster']['component_placement'] = {'actor,env,rollout': str(GPU)}
    for split in ('train', 'eval'):
        item = actual['env'][split]
        old_seed = Path(baseline['env'][split]['seeds_path'])
        seed_relative = old_seed.as_posix().split('/rlinf/envs/robotwin/seeds/', 1)
        assert len(seed_relative) == 2 and '/' not in seed_relative[1]
        item['seeds_path'] = str(ROOT / 'rlinf/envs/robotwin/seeds' / seed_relative[1])
        item['task_config']['save_path'] = str(run / 'robotwin_data')
        old_video = baseline['env'][split]['video_cfg']['video_base_dir']
        assert old_video.startswith(str(BASELINE) + '/')
        item['video_cfg']['video_base_dir'] = str(run) + old_video[len(str(BASELINE)):]
    old_worker_logs = baseline['runner'].get('per_worker_log_path')
    if isinstance(old_worker_logs, str) and old_worker_logs.startswith(str(BASELINE) + '/'):
        actual['runner']['per_worker_log_path'] = str(run) + old_worker_logs[len(str(BASELINE)):]
    clean_env()
    env = environment(run, baseline)
    os.environ.update(env)
    sys.path.insert(0, str(ROOT))
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE = namespace + '_validate'
    cfg = OmegaConf.create(actual)
    try:
        cfg = validate_cfg(cfg)
    finally:
        try:
            cleanup(Cluster.NAMESPACE, pre / 'validation-cleanup.json')
        finally:
            ray.shutdown()
    actual = OmegaConf.to_container(cfg, resolve=True)
    check_budget(actual, settings, key)
    changes = h.diff(baseline, actual)
    identity = {'runner.logger.experiment_name', 'runner.logger.log_path', 'runner.per_worker_log_path',
                'env.train.task_config.save_path', 'env.eval.task_config.save_path',
                'env.train.video_cfg.video_base_dir', 'env.eval.video_cfg.video_base_dir',
                'env.train.seeds_path', 'env.eval.seeds_path', 'algorithm.online_bc.data_path',
                'cluster.component_placement.actor,env,rollout'}
    method = {'algorithm.online_bc.max_success_chunks'} | {
        'algorithm.online_bc.attena.' + field for field in actual['algorithm']['online_bc']['attena']}
    smoke = {'runner.max_epochs', 'runner.val_check_interval', 'runner.save_interval'} if key == 'smoke' else set()
    unexpected = {name: values for name, values in changes.items() if name not in identity | method | smoke}
    assert not unexpected, unexpected
    seeds = {}
    for split in ('train', 'eval'):
        old_path = Path(baseline['env'][split]['seeds_path'])
        new_path = Path(actual['env'][split]['seeds_path'])
        assert sha(old_path) == sha(new_path), (old_path, new_path)
        seeds[split] = {'baseline': str(old_path), 'new': str(new_path), 'sha256': sha(new_path)}
    OmegaConf.save(cfg, pre / 'resolved.yaml', resolve=True)
    # Hydra's config path is the complete resolved snapshot, not a recomposed
    # collection of defaults; its own output stays disabled as in clean BC.
    runtime = run / 'runtime'
    argv = [PY, '-u', '-B', str(ST / 'launch_ops.py'), 'driver', key,
            '--config-path', str(runtime), '--config-name', 'resolved',
            'hydra.run.dir=.', 'hydra.output_subdir=null', 'hydra.job.chdir=false',
            'hydra/job_logging=stdout']
    save(pre / 'argv.json', argv)
    save(pre / 'environment.json', env)
    (pre / 'command.txt').write_text(shlex.join(argv) + '\n')
    (pre / 'command.sh').write_text('#!/usr/bin/env bash\nexec ' + shlex.join(argv) + '\n')
    (pre / 'wrapper.sh').write_text('''#!/usr/bin/env bash
set +e
runtime=$1
ulimit -n 4096 || exit 91
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s SECONDSs bash "$runtime/command.sh" > "$runtime/driver.log" 2>&1 &
child=$!
printf '%s\\n' "$child" > "$runtime/timeout.pid"
wait "$child"
rc=$?
printf '%s\\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
'''.replace('SECONDS', str(cfg_spec['timeout'])))
    for filename in ('wrapper.sh', 'command.sh'):
        subprocess.run(['bash', '-n', str(pre / filename)], check=True)
    save(pre / 'baseline-config.json', baseline)
    save(pre / 'config_diff.json', changes)
    contract = {'time': now(), 'key': key, 'root': str(ROOT), 'run': str(run), 'namespace': namespace,
        'source_head': head, 'source_hashes': hashes, 'base': BASE, 'branch': BRANCH,
        'baseline_run': str(BASELINE), 'baseline_resolved_sha256': sha(baseline_path),
        'physical_gpus': [GPU], 'rounds': cfg_spec['rounds'], 'attempts_per_round': 4,
        'adam_per_round': 5, 'training_attempts': cfg_spec['rounds'] * 4,
        'max_adam_updates': cfg_spec['rounds'] * 5, 'global_batch': 1024, 'micro_batch': 32,
        'denoise_steps': 10, 'chunk_size': 50, 'learning_rate': 2.5e-5, 'length_filter': 3,
        'signal': 'Per-arm target TCP translation/rotation motion, combined as left + right',
        'attena': actual['algorithm']['online_bc']['attena'], 'asset_hashes': assets,
        'settings_sha256': sha(ST / 'attena-launch-settings.json'),
        'eval_interval': cfg_spec['eval'], 'eval_episodes_per_check': 32,
        'save_interval': cfg_spec['save'], 'timeout_seconds': cfg_spec['timeout'], 'seeds': seeds,
        'resolved_sha256': sha(pre / 'resolved.yaml'),
        'prepared_hashes': {name: sha(pre / name) for name in ('resolved.yaml', 'argv.json', 'environment.json', 'command.sh', 'wrapper.sh')},
        'fresh_model_and_empty_pool': True, 'unexpected_differences': unexpected,
        'baseline_confounds': ['Historical clean BC length filtering was off; this run enables <=3 chunks']}
    save(pre / 'contract.json', contract, exclusive=True)
    print(json.dumps({'contract': contract, 'config_diff': changes}), flush=True)


def smoke_gate(hashes, assets):
    _, run, _, pre = select('smoke')
    verification = read(ST / 'smoke-verification.json')
    assert verification['passed'] and verification['checkpoint_restore_passed']
    assert (run / 'runtime/exit_code.txt').read_text().strip() == '0'
    smoke = read(pre / 'contract.json')
    assert smoke['source_hashes'] == hashes and smoke['asset_hashes'] == assets
    return {'verification_sha256': sha(ST / 'smoke-verification.json'), 'run': str(run),
            'passed': True, 'checkpoint_restore_passed': True}


def launch(key):
    import ray
    cfg_spec, run, namespace, pre = select(key)
    head, hashes = source_check()
    _, assets = settings_check()
    contract = read(pre / 'contract.json')
    assert contract['source_head'] == head and contract['source_hashes'] == hashes
    assert contract['asset_hashes'] == assets and contract['unexpected_differences'] == {}
    assert sha(ST / 'attena-launch-settings.json') == contract['settings_sha256']
    for name, expected in contract['prepared_hashes'].items():
        assert sha(pre / name) == expected, name
    assert sha(BASELINE / 'runtime/resolved.yaml') == contract['baseline_resolved_sha256']
    assert not run.exists() and run.parent.resolve() == RESULTS.resolve()
    gate = smoke_gate(hashes, assets) if key == 'formal' else None
    devices = h.gpu_snapshot()
    empty_gpu(devices)
    protected = protected_processes(devices)
    # One bounded capacity check before starting, including the saved smoke model.
    assert shutil.disk_usage('/home').free > (120 if key == 'formal' else 20) * 1024**3
    available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')).split()[1])
    assert available > 200 * 1024**2
    ray.init(address=RAY_ADDRESS, namespace=namespace + '_prelaunch', logging_level='ERROR')
    try:
        assert not any(row['namespace'] == namespace for row in ray.util.list_named_actors(all_namespaces=True))
    finally:
        ray.shutdown()
    save(ST / ('launch-' + key + '-attempt.json'), {'time': now(), 'run': str(run), 'namespace': namespace}, exclusive=True)
    run.mkdir()
    runtime = run / 'runtime'
    shutil.copytree(pre, runtime)
    (runtime / 'source-head.txt').write_text(head + '\n')
    save(runtime / 'protected-before.json', {str(k): v for k, v in protected.items()})
    save(runtime / 'gpu-before.json', devices)
    clean_env()
    env = os.environ.copy()
    env.update(read(runtime / 'environment.json'))
    with (runtime / 'wrapper.log').open('xb') as stream:
        process = subprocess.Popen(['bash', str(runtime / 'wrapper.sh'), str(runtime)], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    wrapper = h.proc(process.pid)
    (runtime / 'wrapper.pid').write_text(str(process.pid) + '\n')
    h.assert_processes_unchanged(protected)
    receipt = {'time': now(), 'key': key, 'root': str(ROOT), 'run': str(run), 'namespace': namespace,
        'physical_gpus': [GPU], 'source_head': head, 'wrapper': wrapper,
        'protected_processes': list(protected.values()), 'protected_unchanged_at_launch': True,
        'smoke_gate': gate}
    save(ST / ('launch-' + key + '-receipt.json'), receipt, exclusive=True)
    print(json.dumps(receipt), flush=True)


def driver(key):
    _, run, namespace, _ = select(key)
    sys.path.insert(0, str(ROOT))
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE = namespace
    runtime = run / 'runtime'
    save(runtime / 'driver-identity.json', {**h.proc(os.getpid()), 'namespace': namespace}, exclusive=True)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    sys.argv = [str(ROOT / 'examples/embodiment/train_embodied_agent.py'), *sys.argv[3:]]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        import ray
        try:
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            assert Cluster.NAMESPACE == namespace
            cleanup(namespace, runtime / 'owned-cleanup.json')
        finally:
            ray.shutdown()


def status(key):
    """One read-only health snapshot; never wait for another round or alter a job."""
    import yaml
    from ray.util.state import list_actors
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _, run, namespace, pre = select(key)
    runtime = run / 'runtime'
    rows = list_actors(address='http://127.0.0.1:8266', detail=True,
        filters=[('state', '=', 'ALIVE')], limit=2000, timeout=15)
    rows = [row.asdict() if hasattr(row, 'asdict') else dict(row) for row in rows]
    devices = h.gpu_snapshot()
    report = {'time': now(), 'key': key, 'run': str(run), 'namespace': namespace,
        'gpu6': devices[GPU], 'actors': [{k: row.get(k) for k in ('name', 'ray_namespace', 'job_id', 'pid', 'state')}
                                      for row in rows if row.get('ray_namespace') == namespace]}
    for name in ('started_at.txt', 'finished_at.txt', 'exit_code.txt', 'driver-identity.json', 'owned-cleanup.json'):
        if (runtime / name).exists():
            report[name] = (runtime / name).read_text()
    if (runtime / 'driver-identity.json').exists():
        expected = read(runtime / 'driver-identity.json')
        try:
            actual = h.proc(expected['pid'])
            report['driver_identity_unchanged'] = all(actual[k] == expected[k] for k in ('pid', 'uid', 'start'))
        except FileNotFoundError:
            report['driver_identity_unchanged'] = False
    if (runtime / 'driver.log').exists():
        with (runtime / 'driver.log').open('rb') as stream:
            stream.seek(max(0, (runtime / 'driver.log').stat().st_size - 12000))
            tail = stream.read().decode(errors='replace')
        report['log_tail'] = tail
        report['fatal_indicators_in_tail'] = [s for s in ('Traceback (most recent call last)', 'CUDA out of memory',
            'Segmentation fault', 'Exception occurred while running', 'Error executing job with overrides') if s in tail]
    actual_path = run / 'tensorboard/config.yaml'
    if actual_path.exists():
        actual = yaml.safe_load(actual_path.read_text())
        expected = yaml.safe_load((pre / 'resolved.yaml').read_text())
        report['actual_config_diff'] = h.diff(expected, actual)
        report['actual_vs_clean_baseline_diff'] = h.diff(read(pre / 'baseline-config.json'), actual)
    if (run / 'tensorboard').exists():
        ea = EventAccumulator(str(run / 'tensorboard'), size_guidance={'scalars': 0})
        ea.Reload()
        data = {tag: ea.Scalars(tag) for tag in ea.Tags()['scalars']}
        report['latest_scalars'] = {tag: {'raw_step': pts[-1].step, 'round': pts[-1].step + 1, 'value': pts[-1].value}
                                    for tag, pts in data.items() if pts}
        report['scalar_counts'] = {tag: len(pts) for tag, pts in data.items()}
        report['scalars_finite'] = all(math.isfinite(pt.value) for pts in data.values() for pt in pts)
    report['protected_processes_status'] = []
    protected_path = runtime / 'protected-before.json'
    if protected_path.exists():
        for expected in read(protected_path).values():
            try:
                actual = h.proc(expected['pid'])
                unchanged = all(actual[k] == expected[k] for k in ('pid', 'uid', 'start'))
            except FileNotFoundError:
                unchanged = False
            # An unrelated job can naturally finish later; report identity instead
            # of turning a stale launch-time liveness condition into an assertion.
            report['protected_processes_status'].append({'pid': expected['pid'], 'identity_still_present': unchanged})
    save(ST / ('status-' + key + '.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    identity_check()
    assert len(sys.argv) >= 3 and sys.argv[1] in ('prepare', 'launch', 'driver', 'status') and sys.argv[2] in SPECS
    globals()[sys.argv[1]](sys.argv[2])
