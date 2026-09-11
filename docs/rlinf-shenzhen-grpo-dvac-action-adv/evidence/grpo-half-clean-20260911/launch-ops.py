"""Exact-target clean GRPO half-sampling phases; batch 512 launch requires explicit approval receipt."""

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

ST = Path('/data/chenyiteng/results/server-maintenance-20260911/grpo-half-clean')
OLD_ST = Path('/data/chenyiteng/results/server-maintenance-20260908/adv-new')
EXPECTED_HEAD = '1d015a2aa03ec8132d8207ba47a2be3dbe1d9591'
BASE = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf')
ROOT = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-clean-half-20260911')
BASE_HEAD = '1d015a2aa03ec8132d8207ba47a2be3dbe1d9591'
BRANCH = 'codex/sz-pi05-grpo-clean-half-20260911'
RUN = Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-clean-half128-formal200-phys45-20260911-v1')
PRE = ST / 'prepared'
PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
NAMESPACE = 'RLinf_grpo_clean_half128_formal45_20260911'
RAY_ADDRESS = '172.17.0.1:6389'
GPUS = (4, 5)


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


def assert_empty(devices):
    for gpu in GPUS:
        assert gpu in (4, 5)
        state = devices[gpu]
        assert not state['processes'] and state['memory_mib'] < 128 and state['utilization'] == 0, (gpu, state)


def protected_processes(devices):
    return {entry['pid']: entry for gpu in (0, 1, 2, 3, 6, 7)
            for entry in devices[gpu]['processes']}


def assert_processes_unchanged(protected):
    for pid, expected in protected.items():
        actual = proc(pid)
        # Ray changes its process title as methods run; identity is PID/UID/start.
        assert all(actual[key] == expected[key] for key in ('pid', 'uid', 'start')), {
            'expected': expected, 'actual': actual}


def create():
    assert not ROOT.exists() and not RUN.exists() and not PRE.exists()
    assert git(BASE, 'rev-parse', 'HEAD') == BASE_HEAD
    attempt('create')
    subprocess.run(['git', '-C', str(BASE), 'worktree', 'add', '-b', BRANCH, str(ROOT), BASE_HEAD], check=True)
    save(ST / 'create-receipt.json', {'time': now(), 'root': str(ROOT), 'branch': BRANCH, 'base': BASE_HEAD})
    print((ST / 'create-receipt.json').read_text())


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


def prepare():
    import yaml
    import ray
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    PRE.mkdir(exist_ok=True)
    base = json.loads((OLD_ST / 'source-preflight.json').read_text())
    oldcfg = yaml.safe_load(base['runtime']['runtime-resume100-to200/resolved.yaml'])
    (PRE / 'baseline-resolved.yaml').write_text(base['runtime']['runtime-resume100-to200/resolved.yaml'])
    (PRE / 'baseline-start-resolved.yaml').write_text(base['runtime']['runtime/resolved.yaml'])
    source_check()
    for rel in ('runtime/command.txt', 'runtime/resolved.yaml', 'runtime-resume100-to200/command.txt', 'runtime-resume100-to200/resolved.yaml', 'runtime-resume100-to200/environment.json'):
        live = Path(oldcfg['runner']['logger']['log_path']) / rel
        assert live.read_text() == base['runtime'][rel], rel
    old_run = oldcfg['runner']['logger']['log_path']
    old_exp = oldcfg['runner']['logger']['experiment_name']
    experiment = 'pi05_grpo_clean_half128_formal200_phys45_20260911_v1'

    def replace(value):
        return value.replace(str(BASE), str(ROOT)).replace(old_run, str(RUN)).replace(old_exp, experiment)

    env = {key: replace(value) for key, value in
           json.loads(base['runtime']['runtime-resume100-to200/environment.json']).items()}
    assert env['RAY_ADDRESS'] == RAY_ADDRESS
    os.environ.update(env)
    sys.path.insert(0, str(ROOT))
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    tokens = shlex.split(replace(base['runtime']['runtime-resume100-to200/command.txt']))
    overrides = {
        'cluster.component_placement': r'{actor\, env\, rollout:"4,5"}',
        'runner.max_steps': '200', 'runner.resume_dir': 'null',
        'runner.val_check_interval': '5', 'runner.save_interval': '10',
        'actor.group_name': 'ActorGroup_cleanhalf_formal45',
        'rollout.group_name': 'RolloutGroup_cleanhalf_formal45',
        'env.group_name': 'EnvGroup_cleanhalf_formal45',
        'algorithm.dvac_gradient_weighting.mode': 'off',
        'env.train.rollout_epoch': '2', 'actor.global_batch_size': '512',
    }
    for key, value in overrides.items():
        positions = [index for index, token in enumerate(tokens) if token.startswith(key + '=')]
        assert len(positions) <= 1
        if positions:
            tokens[positions[0]] = key + '=' + value
        else:
            tokens.append(key + '=' + value)
    config_index = tokens.index('--config-name')
    Cluster.NAMESPACE = NAMESPACE + '_validate'
    with initialize_config_dir(version_base='1.1', config_dir=str(ROOT / 'examples/embodiment/config')):
        cfg = compose(config_name=tokens[config_index + 1], overrides=tokens[config_index + 2:])
        try:
            cfg = validate_cfg(cfg)
        finally:
            try:
                cleanup_namespace(Cluster.NAMESPACE, PRE / 'validation-cleanup-v3.json')
            finally:
                ray.shutdown()
    actual = OmegaConf.to_container(cfg, resolve=True)
    changes = diff(oldcfg, actual)
    identity = {'runner.logger.experiment_name', 'runner.logger.log_path', 'runner.per_worker_log_path',
                'env.train.task_config.save_path', 'env.eval.task_config.save_path',
                'env.train.video_cfg.video_base_dir', 'env.eval.video_cfg.video_base_dir',
                'env.train.seeds_path', 'env.eval.seeds_path',
                'actor.group_name', 'rollout.group_name', 'env.group_name',
                'cluster.component_placement.actor, env, rollout', 'algorithm.dvac_gradient_weighting.output_dir'}
    initialization = {'runner.resume_dir'}
    defaults = {'runner.per_worker_log', 'runner.weight_sync_interval',
                'runner.overlap_env_bootstrap', 'cluster.tracer.enable'}
    unexpected = {key: value for key, value in changes.items()
                  if key not in identity | initialization | defaults | {'env.train.rollout_epoch', 'actor.global_batch_size'}}
    assert not unexpected, unexpected
    assert cfg.actor.micro_batch_size == 32 and cfg.actor.global_batch_size == 512
    assert cfg.algorithm.update_epoch == 2 and cfg.algorithm.group_size == 8
    assert cfg.env.train.total_num_envs == 64 and cfg.env.train.rollout_epoch == 2
    assert cfg.env.train.max_episode_steps == 200 and cfg.env.train.max_steps_per_rollout_epoch == 200
    assert cfg.runner.max_steps == 200 and cfg.runner.max_epochs == oldcfg['runner']['max_epochs']
    assert cfg.runner.resume_dir is None and not cfg.runner.only_eval
    assert cfg.algorithm.adv_type == 'grpo' and cfg.algorithm.logprob_type == 'chunk_level'
    assert cfg.algorithm.filter_rewards and cfg.algorithm.normalize_advantages
    assert cfg.actor.optim.lr == oldcfg['actor']['optim']['lr']
    assert cfg.actor.model.num_steps == 10 and cfg.actor.model.openpi.num_steps == 10
    assert cfg.actor.model.openpi.noise_level == 0.5
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.algorithm.dvac_gradient_weighting.mode == 'off'
    assert not OmegaConf.select(cfg, 'algorithm.prism.enabled', default=False)
    # Execute the original actor's exact minibatch assertion on the intended
    # shape, without constructing model workers or allocating a GPU.
    tree = ast.parse((ROOT / 'rlinf/workers/actor/embodied_fsdp_actor_worker.py').read_text())
    check = next(node for node in ast.walk(tree) if isinstance(node, ast.Assert)
                 and 'rollout_size % batch_size_per_rank' in ast.unparse(node.test))
    expr = compile(ast.Expression(check.test), '<original actor batch assertion>', 'eval')
    world = 2
    chunks = cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch * (cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks)
    per_rank = chunks // world
    assert per_rank == 256 and chunks == 512
    old_ok = eval(expr, {}, {'rollout_size': per_rank, 'batch_size_per_rank': 1024 // world})
    new_ok = eval(expr, {}, {'rollout_size': per_rank, 'batch_size_per_rank': 512 // world})
    assert not old_ok and new_ok
    assert chunks // cfg.actor.global_batch_size * cfg.algorithm.update_epoch == 2
    assert cfg.actor.global_batch_size // world // cfg.actor.micro_batch_size == 8
    save(PRE / 'batch-compatibility.json', {'time': now(), 'source_assertion': ast.unparse(check),
         'chunk_slots_global': chunks, 'chunk_slots_per_rank': per_rank,
         'global_1024_passes': old_ok, 'global_512_passes': new_ok,
         'micro_batches_per_rank_per_update': 8, 'optimizer_calls_per_round': 2,
         'groups_global': 16, 'group_size': 8, 'filtering': 'binary groups use mask; tensor slots stay fixed'})
    assert list(actual['cluster']['component_placement'].values()) == ['4,5']
    for split in ('train', 'eval'):
        old_seed = Path(oldcfg['env'][split]['seeds_path'])
        new_seed = Path(actual['env'][split]['seeds_path'])
        assert hashlib.sha256(old_seed.read_bytes()).digest() == hashlib.sha256(new_seed.read_bytes()).digest()
    assert not RUN.exists()
    OmegaConf.save(cfg, PRE / 'resolved.yaml', resolve=True)
    # Dedicated driver changes only this process's namespace; workers read the
    # propagated CLUSTER_NAMESPACE. SYS_NAME and the shared Ray service stay intact.
    driver_tokens = [tokens[0], '-B', str(ST / 'ops.py'), 'formal_driver', *tokens[2:]]
    (PRE / 'command.txt').write_text(shlex.join(driver_tokens) + '\n')
    (PRE / 'baseline-command-derived.txt').write_text(shlex.join(tokens) + '\n')
    save(PRE / 'argv.json', driver_tokens)
    save(PRE / 'environment.json', env)
    wrapper = '''#!/usr/bin/env bash
set +e
runtime=$1
ulimit -n 4096 || exit 91
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 345600s bash "$runtime/command.sh" > "$runtime/driver.log" 2>&1
rc=$?
printf '%s\\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
'''
    (PRE / 'wrapper.sh').write_text(wrapper)
    (PRE / 'command.sh').write_text('#!/usr/bin/env bash\nexec ' + shlex.join(driver_tokens) + '\n')
    for name in ('wrapper.sh', 'command.sh'):
        subprocess.run(['bash', '-n', str(PRE / name)], check=True)
    save(PRE / 'config_diff.json', changes)
    save(PRE / 'contract.json', {'time': now(), 'root': str(ROOT), 'run': str(RUN),
         'base': BASE_HEAD, 'physical_gpus': list(GPUS), 'namespace': NAMESPACE,
         'rounds': 200, 'trajectories_per_round': 128, 'group_size': 8, 'update_epochs': 2,
         'global_batch': 512, 'micro_batch': 32, 'denoise_steps': 10, 'noise': 0.5,
         'chunk_horizon': 50, 'episode_limit': 200, 'learning_rate': cfg.actor.optim.lr,
         'dvac': 'off', 'prism': 'absent',
         'save_interval': 10, 'eval_interval': 5, 'eval_episodes_per_check': 32, 'timeout_seconds': 345600,
         'training_attempts': 25600, 'max_optimizer_calls': 400, 'checkpoint_generations': 20,
         'initialization': 'original model, no resume', 'source_head': EXPECTED_HEAD,
         'process_nofile_limit': 4096, 'formal_replacement': True})
    print(json.dumps({'config_diff': changes, 'contract': json.loads((PRE / 'contract.json').read_text())}), flush=True)






def launch():
    approval = json.loads((ST / 'user-approved-contract.json').read_text())
    assert approval['global_batch_size'] == 512 and approval['rollout_epoch'] == 2
    assert approval['root_resource_release_confirmed'] is True
    assert approval['contract_sha256'] == hashlib.sha256((PRE / 'contract.json').read_bytes()).hexdigest()
    source = source_check()
    assert git(ROOT, 'rev-parse', 'HEAD') == source['head'] and not git(ROOT, 'status', '--porcelain')
    assert git(ROOT, 'branch', '--show-current') == BRANCH
    assert not RUN.exists()
    devices = gpu_snapshot()
    assert_empty(devices)
    protected = protected_processes(devices)
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


def formal_driver():
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







def source_check():
    assert git(ROOT, 'rev-parse', 'HEAD') == EXPECTED_HEAD
    assert git(ROOT, 'branch', '--show-current') == BRANCH and not git(ROOT, 'status', '--porcelain')
    files = ['rlinf/config.py', 'rlinf/workers/actor/embodied_fsdp_actor_worker.py',
             'rlinf/workers/rollout/hf/huggingface_worker.py', 'rlinf/runners/embodied_runner.py',
             'rlinf/models/embodiment/openpi/openpi_action_model.py',
             'examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml']
    hashes = {}
    for path in files:
        data = (ROOT / path).read_bytes()
        base_data = subprocess.check_output(['git', '-C', str(BASE), 'show', BASE_HEAD + ':' + path])
        assert data == base_data, path
        hashes[path] = hashlib.sha256(data).hexdigest()
    value = {'time': now(), 'head': EXPECTED_HEAD, 'base': BASE_HEAD, 'branch': BRANCH,
             'source_hashes': hashes, 'production_changes': [], 'clean_source_identical': True}
    save(ST / 'source-receipt.json', value)
    return value


def status():
    import urllib.request
    import yaml
    runtime = RUN / 'runtime'
    expected = yaml.safe_load((PRE / 'resolved.yaml').read_text())
    actual_path = RUN / 'tensorboard/config.yaml'
    actual_diff = diff(expected, yaml.safe_load(actual_path.read_text())) if actual_path.exists() else None
    rows = []
    with urllib.request.urlopen('http://127.0.0.1:8266/api/v0/actors?limit=2000&detail=1&filter_keys=state&filter_predicates=%3D&filter_values=ALIVE', timeout=15) as resp:
        payload = json.load(resp)
    rows = payload.get('data', {}).get('result', {}).get('result', [])
    own = [r for r in rows if r.get('ray_namespace') == NAMESPACE]
    driver = json.loads((runtime / 'driver-identity.json').read_text()) if (runtime / 'driver-identity.json').exists() else None
    if driver:
        p = proc(driver['pid'])
        assert all(p[k] == driver[k] for k in ('pid', 'uid', 'start'))
    protected = json.loads((runtime / 'protected-before.json').read_text())
    assert_processes_unchanged({int(k): v for k, v in protected.items()})
    log = (runtime / 'driver.log').read_text(errors='replace') if (runtime / 'driver.log').exists() else ''
    fatal = [line for line in log.splitlines() if any(k in line for k in ('Traceback (most recent call last)', 'CUDA out of memory', 'AssertionError:', 'RuntimeError:'))]
    value = {'time': now(), 'run': str(RUN), 'namespace': NAMESPACE, 'driver': driver,
             'own_actors': own, 'actual_config_diff': actual_diff, 'fatal_lines': fatal[-12:],
             'protected_processes_unchanged': True, 'gpus': gpu_snapshot(),
             'driver_log_tail': log[-10000:], 'exit_code': (runtime / 'exit_code.txt').read_text().strip() if (runtime / 'exit_code.txt').exists() else None}
    save(ST / 'startup-health.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    assert os.getuid() == 1003 and pwd.getpwuid(os.getuid()).pw_name == 'chenyiteng'
    {'create': create, 'prepare': prepare, 'launch': launch, 'formal_driver': formal_driver, 'status': status}[sys.argv[1]]()
