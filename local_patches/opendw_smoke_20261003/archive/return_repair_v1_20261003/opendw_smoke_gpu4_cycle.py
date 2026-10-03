"""Borrow only current SZ3 next-six GPU4, then restore its exact RLT config.

This is a cycle component for a separately supervised smoke owner. The owner must
call resume in its finally/recovery path with an exact released-process receipt.
It does not launch the smoke, alter shared Ray, or signal GPU5-7 owners.
All commands must use the current RLT Python on SZ3. Prepare is read-only toward
the original run and saves an independent cycle plan; no missing-CP fresh restart.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

ROOT = Path('/data/chenyiteng')
SOURCE_PLAN = ROOT / 'deployment-20261002/rlt-next6-beat_block_hammer/plan.json'
RECOVERY = ROOT / 'deployment-20261003/rlt-step-timeout-recovery-v1'
WATCH = ROOT / 'deployment-20260927/rlt-six-task-watch/plan.json'
EXPECTED_HEAD = '2f484040dbf078fcb6bfcc1da8a300fbc625f87b'
SCRIPT_NAME = 'opendw_smoke_gpu4_cycle.py'
H = None
ST = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graphics_scope_spec(manifest_path):
    path = Path(manifest_path).resolve(strict=True)
    assert path.is_relative_to(ROOT.resolve()) and path.stat().st_uid == 20001
    tools = path.parent.parent/('scope_tools_v5' if path.parent.name == 'gpu4-scope-v5' else 'scope_tools_v4')
    assert tools.is_dir(), 'Expected the separately reviewed scope_tools_v4'
    files = [path, tools/'gpu_scope_prepare.py', tools/'gpu_scope_runtime.py']
    spec = {'manifest_path': str(path), 'tools_dir': str(tools),
            'files_sha256': {str(p): sha(p) for p in files}}
    _, manifest = load_graphics_scope(spec)
    spec['files_sha256'].update({manifest[k+'_path']: manifest[k+'_sha256']
                                for k in ('marker', 'profile', 'bootstrap', 'runtime')})
    spec.update(gpu_uuid=manifest['gpu_uuid'], commname=manifest['commname'], token=manifest['token'])
    return spec


def load_graphics_scope(spec):
    for name, expected in spec['files_sha256'].items():
        assert sha(name) == expected, 'Graphics scope dependency changed: '+name
    tools = Path(spec['tools_dir'])
    assert tools.name in ('scope_tools_v4', 'scope_tools_v5') and tools.resolve().is_relative_to(ROOT.resolve())
    cached = sys.modules.get('gpu_scope_runtime')
    if cached is not None:
        assert sha(cached.__file__) == sha(tools/'gpu_scope_runtime.py'), 'Different scope runtime already imported'
    sys.path.insert(0, str(tools))
    helper = import_file('opendw_return_graphics_scope', tools/'gpu_scope_prepare.py')
    manifest = helper.read_manifest(spec['manifest_path'])
    assert manifest['schema'] == 3 and manifest['profile_feature'] == 'commname'
    assert manifest['title_api'] == 'ray._raylet.setproctitle'
    assert manifest['uid'] == 20001 and manifest['physical_gpu'] == 4
    return helper, manifest


def assert_stage(stage):
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert re.fullmatch(r'[a-zA-Z0-9_-]{6,100}', stage.name), 'Unsafe cycle name'
    assert stage.is_absolute() and stage.resolve().is_relative_to(ROOT.resolve())
    assert not stage.is_symlink()
    if stage.exists():
        assert stage.stat().st_uid == 20001


def source_check(plan):
    assert subprocess.check_output(
        ['git', '-C', plan['repo'], 'rev-parse', 'HEAD'], text=True).strip() == EXPECTED_HEAD == plan['head']
    assert not subprocess.check_output(
        ['git', '-C', plan['repo'], 'diff', '--name-only', 'HEAD', '--', 'rlinf', 'examples'],
        text=True).strip(), 'RLT training source changed'
    for rel, digest in plan['source_sha256'].items():
        assert sha(Path(plan['repo']) / rel) == digest, rel


def load_plan(stage):
    assert_stage(stage)
    plan = read(stage / 'plan.json')
    assert plan['cycle_id'] == stage.name and plan['uid'] == 20001
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['boot_id'], 'Host rebooted'
    assert set(plan['runs']) == {'gpu4'} and plan['runs']['gpu4']['gpus'] == [4]
    assert plan['runs']['gpu4']['task'] == 'beat_block_hammer'
    assert sha(__file__) == plan['script_sha256'], 'Cycle script changed'
    assert sha(plan['helper_path']) == plan['helper_sha256'], 'Frozen checkpoint helper changed'
    assert sha(plan['next_six_ops']) == plan['next_six_ops_sha256'], 'Scoped stop helper changed'
    assert sha(plan['source_plan']) == plan['source_plan_sha256'], 'Original next-six plan changed'
    previous = plan.get('previous_cycle')
    if previous:
        prior = Path(previous['path'])
        assert sha(prior/'plan.json') == previous['plan_sha256'], 'Previous cycle plan changed'
        assert sha(prior/SCRIPT_NAME) == previous['script_sha256'], 'Previous frozen driver changed'
    if plan.get('graphics_scope'):
        load_graphics_scope(plan['graphics_scope'])
    source_check(plan)
    return plan


def protected_snapshot():
    plan = read(RECOVERY / 'plan.json')
    assert set(plan['runs']) == {'gpu5', 'gpu6', 'gpu7'}
    result = {'owner': read(RECOVERY / 'owner-identity.json'), 'drivers': {}}
    assert H.same(result['owner']), 'GPU5-7 recovery owner changed or exited'
    for key, row in plan['runs'].items():
        assert row['gpus'] == [int(key[-1])]
        ident = read(Path(row['run']) / 'runtime/driver-identity.json')
        assert H.same(ident) and ident['namespace'] == row['namespace']
        result['drivers'][key] = ident
    return result


def protected_check(plan):
    expected = plan['protected_gpu5_7']
    assert H.same(expected['owner']), 'Protected owner no longer matches'
    assert all(H.same(x) for x in expected['drivers'].values()), 'Protected driver no longer matches'


def watch_matches(original_run, namespace):
    watch = read(WATCH)
    matches = [r for r in watch['runs'].values() if r['gpus'] == [4]]
    assert len(matches) == 1
    assert matches[0]['run'] == str(original_run) and matches[0]['namespace'] == namespace
    return matches[0]


def require_checkpoint(run, cfg, repo):
    recovery = H.select_recovery(run, cfg, repo)
    assert recovery['mode'] == 'resume_checkpoint' and recovery['checkpoint'] is not None, \
        'GPU4 smoke requires a complete checkpoint; fresh Stage2 is not authorized here'
    return recovery


def prepare(stage, helper_path, expected_driver_pid, previous_cycle=None, graphics_scope_manifest=None):
    from omegaconf import OmegaConf
    assert_stage(stage)
    assert not (stage / 'plan.json').exists(), 'Do not replay prepare'
    source = read(SOURCE_PLAN)
    assert source['head'] == EXPECTED_HEAD and source['task'] == 'beat_block_hammer'
    assert source['uid'] == 20001
    row = dict(source['runs']['clean'])
    assert row['gpus'] == [4] and row['repo'] == source['repo']
    previous = None
    if previous_cycle:
        prior_stage = Path(previous_cycle)
        assert_stage(prior_stage)
        assert prior_stage.resolve() != stage.resolve()
        prior = import_file('previous_opendw_gpu4_cycle', prior_stage/SCRIPT_NAME)
        prior.install_helper(prior_stage)
        prior_plan = prior.load_plan(prior_stage)
        assert (prior_stage/'resumed-dispatched.json').is_file(), 'Previous cycle was not returned'
        prior_status = prior.H.status(prior_stage)
        assert prior_status['all_first_rounds_verified'], 'Previous returned RLT has no verified healthy first round'
        returned = prior_plan['runs']['gpu4']
        assert returned['gpus'] == [4] and returned['task'] == 'beat_block_hammer'
        assert prior_plan['repo'] == source['repo'] and prior_plan['head'] == EXPECTED_HEAD
        row.update(run=returned['new_run'], namespace=returned['namespace'], entry=returned['entry'])
        previous = dict(path=str(prior_stage), plan_sha256=sha(prior_stage/'plan.json'),
                        script_sha256=sha(prior_stage/SCRIPT_NAME), returned_run=row['run'],
                        returned_namespace=row['namespace'], first_round_verified=True)
    original_run = Path(row['run'])
    rt = original_run / 'runtime'
    identity = read(rt / 'driver-identity.json')
    assert identity['pid'] == expected_driver_pid and H.same(identity)
    assert identity['namespace'] == row['namespace']
    argv = (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0')
    assert b'driver' in argv
    if previous:
        assert str(prior_stage/SCRIPT_NAME).encode() in argv
        assert b'gpu4' in argv and argv.count(b'--cycle-dir') == 1 and argv.count(b'--key') == 1
        assert argv[argv.index(b'--key')+1] == b'gpu4'
        assert Path(os.fsdecode(argv[argv.index(b'--cycle-dir')+1])).resolve() == prior_stage.resolve()
    else:
        assert str(source['ops']).encode() in argv
        assert b'clean' in argv and argv.count(b'--stage') == 1
        assert Path(os.fsdecode(argv[argv.index(b'--stage') + 1])).resolve() == SOURCE_PLAN.parent.resolve()
    identity.update(cmdline_sha256=H.proc(identity['pid'])['cmdline_sha256'], match_cmdline=True)
    watch_matches(original_run, row['namespace'])
    cfg = H.config(rt / 'resolved.yaml')
    tb = original_run / 'tensorboard/config.yaml'
    if not tb.is_file():
        tb = original_run / original_run.name / 'tensorboard/config.yaml'
    assert cfg == H.config(tb), 'Actual config differs from runtime config'
    assert cfg['env']['train']['total_num_envs'] == 8
    assert cfg['env']['train']['max_episode_steps'] == cfg['env']['eval']['max_episode_steps'] == 200
    assert cfg['runner']['max_epochs'] == cfg['runner']['max_steps'] == 3000
    assert cfg['cluster']['component_placement'] == {'actor,env,rollout': 4}
    recovery = require_checkpoint(original_run, cfg, source['repo'])
    live = H.actors(source)
    owned = H.active(live, row['namespace'])
    jobs = {r['job_id'] for r in owned}
    assert len(jobs) == 1 and owned
    H.validate_actor_rows(owned, row['namespace'], jobs)
    tree = H.process_tree({identity['pid']} | {r['pid'] for r in owned if r.get('pid')})
    assert {r['pid'] for r in H.gpu_processes([4])} <= set(tree), 'Unrelated GPU4 context'
    new_run = original_run.with_name(original_run.name + '-after-' + stage.name)
    namespace = 'rlt-opendw-return-' + stage.name[-35:] + '-g4'
    assert not new_run.exists() and not H.active(live, namespace)
    newcfg, changes = H.resumed_config(cfg, original_run, new_run, recovery['checkpoint']['path'])
    graphics_scope = graphics_scope_spec(graphics_scope_manifest) if graphics_scope_manifest else None
    stage.mkdir(parents=True, mode=0o700, exist_ok=True)
    pre = stage / 'prepared/gpu4'
    pre.mkdir(parents=True, mode=0o700)
    frozen_helper = stage / 'rlt_checkpoint_lifecycle.py'
    frozen_helper.write_bytes(Path(helper_path).read_bytes())
    frozen_helper.chmod(0o500)
    frozen_script = stage / SCRIPT_NAME
    assert frozen_script.resolve() != Path(__file__).resolve(), 'Prepare from the reviewed source path'
    frozen_script.write_bytes(Path(__file__).read_bytes())
    frozen_script.chmod(0o500)
    OmegaConf.save(OmegaConf.create(cfg), pre / 'original.yaml', resolve=True)
    OmegaConf.save(OmegaConf.create(newcfg), pre / 'resolved.yaml', resolve=True)
    environment = read(rt / 'environment.json')
    assert not any(k in environment for k in H.MASKS)
    environment = {k: v.replace(str(original_run), str(new_run)).replace(row['namespace'], namespace)
                   for k, v in environment.items()}
    if graphics_scope:
        scope_helper, _ = load_graphics_scope(graphics_scope)
        environment.update(scope_helper.environment_fragment(graphics_scope['manifest_path'], environment))
        assert not any(k in environment for k in H.MASKS), 'Discovery driver must retain physical inventory'
    H.save(pre / 'environment.json', environment)
    plan = {k: source[k] for k in ('repo', 'head', 'uid', 'python', 'ray_address', 'ray_dashboard_url')}
    plan.update(cycle_id=stage.name, time=H.now(), host=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                script_sha256=sha(__file__), helper_path=str(frozen_helper), helper_sha256=sha(frozen_helper),
                next_six_ops=source['ops'], next_six_ops_sha256=sha(source['ops']),
                source_plan=str(SOURCE_PLAN), source_plan_sha256=sha(SOURCE_PLAN),
                previous_cycle=previous,
                graphics_scope=graphics_scope,
                source_sha256={rel: sha(Path(source['repo']) / rel) for rel in H.CODE_FILES},
                protected_gpu5_7=protected_snapshot(),
                management_namespace='opendw-g4-rlt-ops-' + stage.name[-30:],
                gpu_processes_before=H.gpu_processes(list(range(8))), runs={})
    plan['runs']['gpu4'] = dict(task='beat_block_hammer', kind='clean', gpus=[4],
        original_run=str(original_run), original_namespace=row['namespace'], original_identity=identity,
        original_jobs=sorted(jobs), original_config_sha256=sha(rt / 'resolved.yaml'),
        new_run=str(new_run), namespace=namespace, entry=row['entry'], recovery=recovery,
        config_changes=changes, dependencies=H.dependency_snapshot(cfg),
        latest_metrics_before=H.latest_metrics(original_run),
        prepared_sha256={n: sha(pre/n) for n in ('original.yaml', 'resolved.yaml', 'environment.json')})
    source_check(plan)
    H.save(stage / 'plan.json', plan)
    H.save(stage / 'prepared.json', {'time': H.now(), 'gpus': [4], 'recovery': recovery,
                                    'config_changes': changes, 'new_run': str(new_run)})
    return {'prepared': True, 'gpu': 4, 'checkpoint_step': recovery['checkpoint']['step']}


def stop(stage):
    from omegaconf import OmegaConf
    plan = load_plan(stage)
    protected_check(plan)
    assert not (stage / 'rlt-stopped.json').exists() and not (stage / 'clean-old-stop-attempt.json').exists(), \
        'Do not replay a stop attempt; inspect its exact receipts'
    row = plan['runs']['gpu4']
    assert H.same(row['original_identity'])
    assert sha(Path(row['original_run']) / 'runtime/resolved.yaml') == row['original_config_sha256']
    watch_matches(row['original_run'], row['original_namespace'])
    bound = dict(run=row['original_run'], namespace=row['original_namespace'], gpus=[4],
                 identity=row['original_identity'])
    selected = H.active(H.actors(plan), bound['namespace'])
    H.validate_actor_rows(selected, bound['namespace'], set(row['original_jobs']))
    tree = H.process_tree({bound['identity']['pid']} | {a['pid'] for a in selected if a.get('pid')})
    assert {r['pid'] for r in H.gpu_processes([4])} <= set(tree), 'Foreign compute or graphics context on GPU4'
    pre = stage / 'prepared/gpu4'
    cfg = H.config(pre / 'original.yaml')
    require_checkpoint(row['original_run'], cfg, plan['repo'])
    ops = import_file('scoped_next_six_ops', plan['next_six_ops'])
    ops.ST = stage
    def checked_stop():
        load_plan(stage)
        protected_check(plan)
        return {**plan, 'old_runs': [bound], 'runs': {'clean': {'gpus': [4]}}}
    ops.checked = checked_stop
    # Include graphics contexts in the established scoped stop's availability checks.
    ops.gpu_pids = lambda gpus: sorted({r['pid'] for r in H.gpu_processes(gpus)})
    ops.stop_old([bound], 'clean-')
    return finalize_stopped(stage)


def finalize_stopped(stage):
    """Recover bookkeeping only after this cycle's precise stop completed."""
    from omegaconf import OmegaConf
    plan = load_plan(stage)
    if (stage/'rlt-stopped.json').exists():
        return {'stopped': True, 'already_recorded': True}
    assert (stage/'clean-old-stop-attempt.json').is_file()
    assert (stage/'clean-old-stopped.json').is_file(), 'Exact scoped stop did not complete'
    row = plan['runs']['gpu4']
    pre = stage / 'prepared/gpu4'
    cfg = H.config(pre / 'original.yaml')
    assert not H.gpu_processes([4])
    assert not H.same(row['original_identity'])
    assert not H.active(H.actors(plan), row['original_namespace'])
    protected_after = {
        'owner_same': H.same(plan['protected_gpu5_7']['owner']),
        'drivers_same': {key: H.same(ident) for key, ident in plan['protected_gpu5_7']['drivers'].items()},
    }
    recovery = require_checkpoint(row['original_run'], cfg, plan['repo'])
    newcfg, changes = H.resumed_config(cfg, row['original_run'], row['new_run'], recovery['checkpoint']['path'])
    OmegaConf.save(OmegaConf.create(newcfg), pre/'resolved.yaml', resolve=True)
    metrics = H.latest_metrics(row['original_run'])
    rounds = metrics.get('env/success_once', {}).get('round')
    frozen = dict(recovery=recovery, resolved_sha256=sha(pre/'resolved.yaml'), config_changes=changes,
                  last_metrics=metrics, collection_rounds_not_restored=None if rounds is None else
                  max(0, rounds-recovery['checkpoint']['step']))
    H.save(stage/'rlt-stopped.json', dict(time=H.now(), cycle_id=stage.name, runs={'gpu4': frozen},
        all_original_drivers_stopped=True, all_original_namespaces_empty=True, gpus_released=[4],
        protected_gpu5_7_after=protected_after, gpu_processes_after=H.gpu_processes(list(range(8)))))
    return {'stopped': True, 'gpu': 4, 'checkpoint_step': recovery['checkpoint']['step']}


def validate_release(stage, receipt_path):
    path = Path(receipt_path)
    assert path.resolve().is_relative_to(ROOT.resolve()) and path.stat().st_uid == 20001
    value = read(path)
    assert value['cycle_id'] == stage.name and value['gpus'] == [4]
    assert value['terminal_status'] in ('completed', 'failed', 'timed_out', 'not_started')
    assert value['all_workers_stopped'] is True
    rows = value['managed_processes']
    assert isinstance(rows, list) and (rows or value['terminal_status'] == 'not_started')
    assert all(r['uid'] == 20001 and not H.same(r) for r in rows), 'Smoke process remains'
    assert not H.gpu_processes([4]), 'Smoke compute or graphics context remains on GPU4'
    return {'path': str(path), 'sha256': sha(path), 'terminal_status': value['terminal_status']}


def update_watch(plan, stage):
    row = plan['runs']['gpu4']
    assert not WATCH.is_symlink() and WATCH.stat().st_uid == 20001
    with (WATCH.parent/'route.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = read(WATCH)
        after = copy.deepcopy(before)
        candidates = [(key, r) for key, r in after['runs'].items() if r['gpus'] == [4]]
        assert len(candidates) == 1
        key, target = candidates[0]
        assert target['run'] in (row['original_run'], row['new_run'])
        if target['run'] == row['new_run']:
            assert target['namespace'] == row['namespace']
            return
        assert target['namespace'] == row['original_namespace']
        H.save(stage/'watch-before.json', before)
        target.update(run=row['new_run'], namespace=row['namespace'])
        H.atomic(WATCH, after)
        H.save(stage/'watch-updated.json', {'time': H.now(), 'key': key, 'target': target})


def guard_names(plan, stage):
    path = ROOT/'security/ray-guard/training_allowlist.json'
    assert not path.is_symlink() and path.stat().st_uid == 20001
    with (path.parent/'recovery-allowlist.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = read(path)
        names = {plan['runs']['gpu4']['namespace']}
        if names <= set(before['namespaces']):
            return
        if not (stage/'allowlist-before.json').exists():
            H.save(stage/'allowlist-before.json', before)
        after = dict(before)
        after['namespaces'] = sorted(set(before['namespaces']) | names)
        H.atomic(path, after)
        H.save(stage/'allowlist-added.json', {'time': H.now(), 'added': sorted(names-set(before['namespaces']))})


def install_helper(stage, helper_path=None):
    global H
    if helper_path is None:
        helper_path = read(stage/'plan.json')['helper_path']
    H = import_file('frozen_rlt_checkpoint_lifecycle', helper_path)
    H.SCRIPT_NAME = SCRIPT_NAME
    H.load_plan = load_plan
    H.source_check = source_check
    H.validate_release = validate_release
    H.update_watch = update_watch
    H.guard_names = guard_names
    original_driver, original_status = H.driver, H.status

    def scoped_driver(target, key):
        plan = load_plan(target)
        spec = plan.get('graphics_scope')
        if not spec:
            return original_driver(target, key)
        scope_helper, manifest = load_graphics_scope(spec)
        assert not any(k in os.environ for k in H.MASKS), 'Do not mask the discovery driver'
        assert Path(os.environ['RLINF_OPENDW_GPU_SCOPE_MANIFEST']).resolve() == Path(spec['manifest_path'])
        assert Path('/proc/self/comm').read_text().strip() == manifest['commname'], 'Driver bootstrap did not run'
        import ray
        original_init = ray.init

        def scope_init(*args, **kwargs):
            runtime = copy.deepcopy(kwargs.get('runtime_env') or {})
            env_vars = dict(runtime.get('env_vars') or {})
            assert not any(k in env_vars for k in H.MASKS), 'Do not mask CPU/discovery Ray actors at job level'
            inherited = dict(os.environ, **env_vars)
            fragment = scope_helper.environment_fragment(spec['manifest_path'], inherited)
            assert not any(k in fragment for k in H.MASKS)
            env_vars.update(fragment)
            runtime['env_vars'] = env_vars
            kwargs['runtime_env'] = runtime
            result = original_init(*args, **kwargs)
            job = ray.get_runtime_context().get_job_id()
            job = job.hex() if hasattr(job, 'hex') else str(job)
            rt = Path(plan['runs'][key]['new_run'])/'runtime'
            receipt = rt/'graphics-scope-job.json'
            if receipt.exists():
                assert read(receipt)['job_id'] == job
            else:
                H.save(receipt, dict(time=H.now(), job_id=job,
                    namespace=plan['runs'][key]['namespace'], driver=H.proc(os.getpid()),
                    manifest_path=spec['manifest_path'], runtime_env_fragment=fragment))
            return result

        ray.init = scope_init
        try:
            return original_driver(target, key)
        finally:
            ray.init = original_init

    def scoped_status(target):
        result = original_status(target)
        plan = load_plan(target)
        if not plan.get('graphics_scope'):
            return result
        scope = return_graphics_status(plan)
        result['graphics_scope'] = scope
        if not scope['verified']:
            result['all_first_rounds_verified'] = False
            result['runs']['gpu4']['first_round_verified'] = False
        return result

    H.driver, H.status = scoped_driver, scoped_status


def return_graphics_status(plan):
    spec = plan['graphics_scope']
    _, manifest = load_graphics_scope(spec)
    row = plan['runs']['gpu4']
    rt = Path(row['new_run'])/'runtime'
    result = dict(required=True, verified=False, manifest_path=spec['manifest_path'],
                  scope_inherited=False, all_owned_contexts_gpu4=False)
    if not (rt/'graphics-scope-job.json').is_file() or not (rt/'driver-identity.json').is_file():
        return dict(result, pending='Return driver/job receipt is not available')
    receipt = read(rt/'graphics-scope-job.json')
    driver = read(rt/'driver-identity.json')
    assert receipt['namespace'] == row['namespace'] and receipt['manifest_path'] == spec['manifest_path']
    if not H.same(driver) or not H.same(receipt['driver']):
        return dict(result, pending='Return driver is not alive with its recorded identity')
    actors = H.active(H.actors(plan), row['namespace'])
    H.validate_actor_rows(actors, row['namespace'], {receipt['job_id']})
    tree = H.process_tree({driver['pid']} | {a['pid'] for a in actors if a.get('pid')})
    owned = [p for p in H.gpu_processes(list(range(8))) if p['pid'] in tree]
    gpu_pids = {p['pid'] for p in owned}
    evidence = []
    for path in Path(manifest['receipts_dir']).glob('*.json'):
        record = read(path)
        identity = tree.get(record.get('pid'))
        if identity and record.get('token') == manifest['token'] and record.get('start_ticks') == identity['start']:
            evidence.append(record)
    checked = {}
    for pid in {driver['pid']} | gpu_pids:
        proc_path = Path('/proc')/str(pid)
        try:
            maps = (proc_path/'maps').read_text()
            marker_loaded = any(len(parts := line.split(maxsplit=5)) == 6 and
                                parts[5] == manifest['marker_path'] for line in maps.splitlines())
            comm = (proc_path/'comm').read_text().strip()
            boot = [e for e in evidence if e['pid'] == pid and e['phase'] == 'bootstrap'
                    and e.get('comm') == manifest['commname'] and e.get('setproctitle_patched') is True
                    and e.get('setproctitle_api') == manifest['title_api'] and e.get('home') == manifest['home']]
            cuda_ok = pid not in gpu_pids or any(e.get('cuda_visible_devices') == '4'
                       and e.get('gpu_uuid') == manifest['gpu_uuid'] for e in boot)
            checked[pid] = dict(identity=tree[pid], marker_loaded=marker_loaded, comm=comm,
                                bootstrap_receipt=bool(boot), cuda_placement_receipt=cuda_ok,
                                verified=marker_loaded and comm == manifest['commname'] and bool(boot) and cuda_ok)
        except OSError as exc:
            checked[pid] = dict(verified=False, error=str(exc))
    renderer = [e for e in evidence if e['phase'] == 'renderer_bound' and e['pid'] in gpu_pids
                and e.get('comm') == manifest['commname'] and e.get('cuda_visible_devices') == '4'
                and e.get('cuda_uuid') == manifest['gpu_uuid'] and e.get('cuda_device_count') == 1
                and e.get('cuda_pci', '').lower()[-10:] == manifest['pci'].lower()[-10:]]
    inherited = bool(gpu_pids and renderer and checked and all(p['verified'] for p in checked.values()))
    in_scope = bool(owned) and all(p['gpu'] == 4 for p in owned)
    result.update(verified=inherited and in_scope, scope_inherited=inherited,
                  all_owned_contexts_gpu4=in_scope, owned_contexts=owned,
                  processes=checked, renderer_receipts=renderer, job_id=receipt['job_id'])
    return result


def main():
    global ST
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cycle-dir', required=True)
    p.add_argument('action', choices=('prepare', 'stop', 'resume', 'status', 'driver'))
    p.add_argument('--helper', help='Existing audited rlt_cycle_sz3.py, prepare only')
    p.add_argument('--expected-driver-pid', type=int, help='Live verified nextsix GPU4 driver PID')
    p.add_argument('--previous-cycle', help='Explicit already-returned GPU4 cycle to borrow again')
    p.add_argument('--graphics-scope-manifest', help='Optional reviewed GPU4 commname scope, prepare only')
    p.add_argument('--key', choices=('gpu4',))
    p.add_argument('--release-receipt')
    args = p.parse_args()
    ST = Path(args.cycle_dir)
    assert_stage(ST)
    install_helper(ST, args.helper if args.action == 'prepare' else None)
    if args.action == 'driver':
        assert args.key == 'gpu4'
        H.driver(ST, 'gpu4')
        return
    if args.action == 'status':
        result = H.status(ST)
        result['gpu_processes'] = H.gpu_processes(list(range(8)))
        if result['all_first_rounds_verified'] and not (ST/'first-round-verified.json').exists():
            H.save(ST/'first-round-verified.json', result)
        print(json.dumps(result, ensure_ascii=False))
        return
    ST.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (ST/'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            assert args.helper and args.expected_driver_pid
            result = prepare(ST, args.helper, args.expected_driver_pid, args.previous_cycle,
                             args.graphics_scope_manifest)
        elif args.action == 'stop':
            result = stop(ST)
        else:
            assert args.release_receipt
            # The resume path remains available even if an unrelated protected
            # job naturally finishes after borrowing; never strand GPU4 for that.
            result = H.resume(ST, args.release_receipt)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
