"""Reborrow the exact RLT jobs returned by a terminated OpenDW owner.

Reuse the reviewed GPU567 stop/return implementation with a fresh checkpoint
cycle. The previous WM owner must have terminated; no live monitor is retired.
GPU4 and GPU567 retain separate repositories and checkpoint contracts.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/data/chenyiteng')
SCRIPT_NAME = 'rlt_returned_cycle.py'
HEADS = {'gpu4': '2f484040dbf078fcb6bfcc1da8a300fbc625f87b',
         'gpu567': 'd48e4a18f560e648635de9bb9a37dc8101ba1697'}
BASE_SHA = 'aceeb78f29bc508bbb7211bd072139450479aecf80647fd52ebc60cba4ea7bf3'
H = None
B = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def owned(path, exists=True):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
    assert not path.is_symlink()
    if exists:
        assert path.exists() and path.stat().st_uid == 20001
    return path


def source_check(plan):
    import subprocess
    assert plan['head'] == HEADS[plan['group']]
    assert subprocess.check_output(['git', '-C', plan['repo'], 'rev-parse', 'HEAD'], text=True).strip() == plan['head']
    assert not subprocess.check_output(['git', '-C', plan['repo'], 'diff', '--name-only', 'HEAD', '--', 'rlinf', 'examples'], text=True).strip()
    for rel, digest in plan['source_sha256'].items():
        assert sha(Path(plan['repo']) / rel) == digest, 'RLT source changed: ' + rel


def parent_complete(owner, child_path, child_module, helper=None):
    owner = owned(owner)
    p = read(owner / 'owner-plan.json')
    f = read(owner / 'final.json')
    terminal = f['terminal_status']
    assert terminal in ('completed', 'failed'), 'Previous WM owner has no supported terminal result'
    if terminal == 'completed':
        assert f.get('error') is None
    else:
        assert isinstance(f.get('error'), dict) and f['error'].get('type') and f['error'].get('error')
    assert f.get('recovery_error') is None and f.get('rlt_borrowed') is True
    assert f.get('rlt_return_dispatched') is True
    assert Path(p['owner_dir']).resolve() == owner.resolve(), 'Wrong parent owner plan'
    identity = read(owner / 'owner-identity.json')
    assert identity['uid'] == 20001 and identity['pid'] > 0 and identity['start'] > 0
    if helper is not None:
        assert not helper.same(identity), 'Previous WM owner is still live'
    combined = owned(p['lifecycle_path'])
    assert read(owner / 'cleanup.json')['all_stopped'] is True
    release = read(owner / 'smoke-release.json')
    assert release['cycle_id'] == combined.name and release['gpus'] == [4, 5, 6, 7]
    assert release['all_workers_stopped'] is True and release['terminal_status'] == terminal
    assert isinstance(release['managed_processes'], list) and release['managed_processes']
    assert all(row['uid'] == 20001 for row in release['managed_processes'])
    if helper is not None:
        assert all(not helper.same(row) for row in release['managed_processes']), 'Previous WM worker is still live'
    assert read(combined / 'return-started.json')['release_sha256'] == sha(owner / 'smoke-release.json')
    children = read(combined / 'plan.json')['children']
    matches = [(key, value) for key, value in children.items()
               if Path(value['path']).resolve() == Path(child_path).resolve()
               and Path(value['module']).resolve() == Path(child_module).resolve()]
    assert len(matches) == 1, 'Previous cycle is not a child of the terminated owner'
    key, frozen = matches[0]
    assert sha(Path(child_path) / 'plan.json') == frozen['plan_sha256']
    assert sha(child_module) == frozen['module_sha256']
    returned = read(combined / 'resumed-dispatched.json')
    assert set(returned['children']) == {'gpu4', 'gpu567'}
    assert read(owner / 'rlt-return-dispatched.json')['result'] == returned['children']
    assert returned['children'][key]['cycle_id'] == Path(child_path).name
    child_return = read(Path(child_path) / 'resumed-dispatched.json')
    child_plan = read(Path(child_path) / 'plan.json')
    assert child_return['cycle_id'] == Path(child_path).name
    assert set(child_return['runs']) == set(child_plan['runs'])
    assert all(child_return['runs'][k]['run'] == row['new_run'] for k,row in child_plan['runs'].items())
    return key, identity, {'owner_plan': str(owner / 'owner-plan.json'),
                          'owner_plan_sha256': sha(owner / 'owner-plan.json'),
                          'final': str(owner / 'final.json'), 'final_sha256': sha(owner / 'final.json'),
                          'identity': str(owner / 'owner-identity.json'),
                          'identity_sha256': sha(owner / 'owner-identity.json'),
                          'cleanup': str(owner / 'cleanup.json'), 'cleanup_sha256': sha(owner / 'cleanup.json'),
                          'release': str(owner / 'smoke-release.json'), 'release_sha256': sha(owner / 'smoke-release.json'),
                          'owner_return': str(owner / 'rlt-return-dispatched.json'),
                          'owner_return_sha256': sha(owner / 'rlt-return-dispatched.json'),
                          'return_started': str(combined / 'return-started.json'),
                          'return_started_sha256': sha(combined / 'return-started.json'),
                          'combined_plan': str(combined / 'plan.json'),
                          'combined_plan_sha256': sha(combined / 'plan.json'),
                          'combined_return': str(combined / 'resumed-dispatched.json'),
                          'combined_return_sha256': sha(combined / 'resumed-dispatched.json'),
                          'child_return': str(Path(child_path) / 'resumed-dispatched.json'),
                          'child_return_sha256': sha(Path(child_path) / 'resumed-dispatched.json')}


def check_return_invocation(identity, previous_stage, previous_module, key):
    argv = [os.fsdecode(x) for x in (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0') if x]
    assert argv.count(str(previous_module)) == 1 and argv.count('--cycle-dir') == 1 and argv.count('--key') == 1
    assert Path(argv[argv.index('--cycle-dir') + 1]).resolve() == Path(previous_stage).resolve()
    assert argv[argv.index('--key') + 1] == key and 'driver' in argv
    launched = read(Path(previous_stage) / (key + '-launched.json'))
    prior = read(Path(previous_stage) / 'plan.json')['runs'][key]
    assert all(launched['identity'][k] == identity[k] for k in ('pid', 'uid', 'start'))
    assert launched['namespace'] == identity['namespace'] == prior['namespace']
    assert launched['run'] == prior['new_run'], 'Return driver is not the recorded launch'


def adopted_parent(owner, child_path, child_module, handoff_intent, helper=None):
    """Accept only the recorded, locked smoke-to-formal handoff; never a generic failure."""
    owner, child_path = owned(owner), owned(child_path)
    intent_path = owned(handoff_intent)
    intent = read(intent_path)
    assert intent['kind'] == 'opendw-formal-adopt-held-rlt'
    assert Path(intent['parent_owner']).resolve() == owner.resolve()
    assert intent['physical_gpus'] == [4, 5, 6, 7]
    plan_path = owner / 'owner-plan.json'
    assert sha(plan_path) == intent['owner_plan_sha256']
    plan = read(plan_path)
    assert Path(plan['owner_dir']).resolve() == owner.resolve()
    combined = owned(plan['lifecycle_path'])
    assert combined.resolve() == Path(intent['lifecycle_path']).resolve()
    assert sha(combined / 'plan.json') == intent['combined_plan_sha256']
    children = read(combined / 'plan.json')['children']
    assert set(children) == {'gpu4', 'gpu567'}
    matches = [key for key, row in children.items() if Path(row['path']).resolve() == child_path.resolve()
               and Path(row['module']).resolve() == Path(child_module).resolve()]
    assert len(matches) == 1
    group = matches[0]
    for key, row in children.items():
        stage = owned(row['path'])
        assert sha(stage / 'plan.json') == row['plan_sha256'] == intent['child_plan_sha256'][key]
        assert sha(owned(row['module'])) == row['module_sha256'] == intent['child_module_sha256'][key]
        assert not (stage / 'resumed-dispatched.json').exists()
        assert not list(stage.glob('*-launch-attempt.json')) and not list(stage.glob('*-launched.json'))
        stopped = read(stage / 'rlt-stopped.json')
        assert stopped['cycle_id'] == stage.name
        assert stopped['all_original_drivers_stopped'] is True and stopped['all_original_namespaces_empty'] is True
        expected = [4] if key == 'gpu4' else [5, 6, 7]
        assert stopped['gpus_released'] == expected
        assert set(stopped['runs']) == {'gpu'+str(g) for g in expected}
    assert not (combined / 'resumed-dispatched.json').exists() and not (combined / 'return-started.json').exists()
    assert not (owner / 'rlt-return-dispatched.json').exists()
    full_stop = read(combined / 'rlt-stopped.json')
    assert full_stop['cycle_id'] == combined.name and full_stop['gpus_released'] == [4, 5, 6, 7]
    assert full_stop['all_original_drivers_stopped'] is True and full_stop['all_original_namespaces_empty'] is True
    assert set(full_stop['runs']) == {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
    identity = read(owner / 'owner-identity.json')
    assert identity['uid'] == 20001
    assert all(identity[k] == intent['owner_identity'][k] for k in ('pid', 'uid', 'start'))
    final = read(owner / 'final.json')
    assert final['rlt_borrowed'] is True and final['rlt_return_dispatched'] is False
    assert not final.get('partial_rlt_recovery_recorded')
    error = final['error']
    interrupted = (final['terminal_status'] == 'failed' and isinstance(error, dict)
        and error['type'] == 'RuntimeError' and error['error'] in {
            'Owner received signal 15', 'Termination requested during launch',
            'Termination requested during RLT borrowing'})
    assert interrupted or (final['terminal_status'] == 'completed' and error is None)
    recovery = final['recovery_error']
    assert recovery['type'] == 'BlockingIOError' and recovery['error'] == '[Errno 11] Resource temporarily unavailable'
    assert read(owner / 'cleanup.json')['all_stopped'] is True
    release = read(owner / 'smoke-release.json')
    assert release['cycle_id'] == combined.name and release['gpus'] == [4, 5, 6, 7]
    assert release['all_workers_stopped'] is True and release['terminal_status'] == final['terminal_status']
    if helper is not None:
        holder = intent['holder_identity']
        assert holder['pid'] == os.getpid() and holder['uid'] == 20001 and helper.same(holder)
        assert not helper.same(identity)
        assert all(row['uid'] == 20001 and not helper.same(row) for row in release['managed_processes'])
        assert not helper.gpu_processes([4, 5, 6, 7])
        # A second independent open must be unable to acquire the held flock.
        with (combined / 'operation.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
                raise AssertionError('Handoff must retain the previous combined operation lock')
    evidence = [intent_path, plan_path, owner / 'owner-identity.json', owner / 'final.json',
                owner / 'cleanup.json', owner / 'smoke-release.json', combined / 'plan.json',
                combined / 'rlt-stopped.json']
    evidence += [Path(row['path']) / 'rlt-stopped.json' for row in children.values()]
    return group, identity, {str(path): sha(path) for path in evidence}


def load_plan(stage):
    stage = owned(stage)
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    p = read(stage / 'plan.json')
    assert p['cycle_id'] == stage.name and p['uid'] == 20001
    assert p['group'] in HEADS and p['head'] == HEADS[p['group']]
    expected = [4] if p['group'] == 'gpu4' else [5, 6, 7]
    assert {k: row['gpus'] for k, row in p['runs'].items()} == {'gpu'+str(g): [g] for g in expected}
    assert p['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert sha(__file__) == p['script_sha256']
    for path, digest in p['frozen_files'].items():
        assert sha(owned(path)) == digest, 'Frozen lifecycle evidence changed: ' + path
    assert not H.same(p['old_owner']), 'Previous WM owner reappeared'
    for row in p['runs'].values():
        for name, digest in row['original_runtime_sha256'].items():
            assert sha(Path(row['original_run']) / 'runtime' / name) == digest
    source_check(p)
    return p


def retired_parent(stage, plan):
    assert not H.same(plan['old_owner'])
    path = stage / 'monitor-retired.json'
    value = {'time': H.now(), 'identity': plan['old_owner'], 'monitor_only': True,
             'reason': 'Previous OpenDW owner already terminated and returned RLT; no signal sent'}
    if path.exists():
        assert read(path)['identity'] == plan['old_owner']
    else:
        H.save(path, value)


def scope_overlay(stage, path, value):
    """Only the exact new return environment files can receive the scope overlay."""
    p = read(stage / 'plan.json')
    allowed = {str((stage / 'prepared' / k / 'environment.json').resolve()) for k in p['runs']}
    allowed |= {str((Path(r['new_run']) / 'runtime/environment.json').resolve()) for r in p['runs'].values()}
    if str(Path(path).resolve()) not in allowed or not p.get('scope_activation'):
        return value
    activation = owned(p['scope_activation'], exists=False)
    if not activation.exists():
        return value
    receipt = read(activation)
    if receipt['status'] == 'rolled_back':
        return value
    assert receipt['status'] == 'active' and receipt['uid'] == 20001
    assert receipt['scope_id'] == p['scope_id']
    fragment = receipt['environment_fragment']
    assert isinstance(fragment, dict) and fragment and not any(k in fragment for k in H.MASKS)
    manifest_key = 'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'
    allowed_keys = {'LD_PRELOAD', 'PYTHONPATH', manifest_key, '__GL_APPLICATION_PROFILE',
                    '__GL_APPLICATION_PROFILE_LOG', 'HOME', 'USER', 'LOGNAME'}
    assert set(fragment) <= allowed_keys
    assert fragment[manifest_key] == p['scope_manifest']
    assert sha(owned(p['scope_manifest'])) == receipt['manifest_sha256']
    runtime = owned(receipt['runtime_path'])
    assert sha(runtime) == receipt['runtime_sha256']
    scope = load('returned_scope_verify_' + stage.name, runtime)
    manifest = scope.read_manifest(p['scope_manifest'])
    result = dict(value)
    legacy_key = 'RLINF_OPENDW_GPU_SCOPE_MANIFEST'
    old_bootstrap, old_marker = None, None
    if result.get(legacy_key):
        legacy = p['legacy_scopes'][result[legacy_key]]
        assert sha(owned(result[legacy_key])) == legacy['manifest_sha256']
        old_bootstrap, old_marker = legacy['bootstrap_dir'], legacy['marker_path']
        result.pop(legacy_key)
    bootstrap = str(Path(manifest['bootstrap_path']).parent)
    marker = manifest['marker_path']
    assert fragment['PYTHONPATH'].split(':')[0] == bootstrap
    assert fragment['LD_PRELOAD'].replace(' ', ':').split(':')[0] == marker
    # Fragments may originate from WM envs; retain each RLT repo's path tail.
    old_paths = [x for x in result.get('PYTHONPATH', '').split(':') if x and x not in (old_bootstrap, bootstrap)]
    old_preloads = [x for x in result.get('LD_PRELOAD', '').replace(' ', ':').split(':') if x and x not in (old_marker, marker)]
    result['PYTHONPATH'] = ':'.join([bootstrap, *old_paths])
    result['LD_PRELOAD'] = ':'.join([marker, *old_preloads])
    for key in ('HOME', 'USER', 'LOGNAME'):
        if key in fragment:
            assert result.get(key) in (None, fragment[key]), 'RLT account environment changed: ' + key
            result[key] = fragment[key]
    result[manifest_key] = fragment[manifest_key]
    result['__GL_APPLICATION_PROFILE'] = '1'
    if '__GL_APPLICATION_PROFILE_LOG' in fragment:
        result['__GL_APPLICATION_PROFILE_LOG'] = fragment['__GL_APPLICATION_PROFILE_LOG']
    assert fragment.get('__GL_APPLICATION_PROFILE', '1') == '1'
    assert not any(k in result for k in H.MASKS)
    return result


def install_helper(stage, helper_path=None):
    global B, H
    stage = Path(stage)
    p = read(stage / 'plan.json')
    assert sha(p['base_module']) == BASE_SHA
    B = load('returned_base_' + stage.name, p['base_module'])
    B.GPUS = [4] if p['group'] == 'gpu4' else [5, 6, 7]
    B.KEY_GPUS = {'gpu'+str(g): [g] for g in B.GPUS}
    B.SCRIPT_NAME = SCRIPT_NAME
    B.load_plan, B.source_check, B.retire_monitor = load_plan, source_check, retired_parent
    B.install_helper(stage, helper_path)
    H = B.H
    original_read = H.read
    H.read = lambda path: scope_overlay(stage, path, original_read(path))


def prepare(stage, previous_stage, previous_module, parent_owner, base_module,
            scope_activation=None, scope_manifest=None, scope_id=None):
    from omegaconf import OmegaConf
    stage, previous_stage = owned(stage, False), owned(previous_stage)
    assert not (stage / 'plan.json').exists()
    previous_module, base_module = owned(previous_module), owned(base_module)
    assert sha(base_module) == BASE_SHA
    # Check the parent's frozen module/plan before executing that module.
    parent_complete(parent_owner, previous_stage, previous_module)
    prior = load('returned_prior_prepare', previous_module)
    prior.install_helper(previous_stage)
    prior_plan = prior.load_plan(previous_stage)
    helper = prior.H
    group, old_owner, evidence = parent_complete(parent_owner, previous_stage, previous_module, helper)
    assert prior_plan['head'] == HEADS[group]
    live = helper.actors(prior_plan)
    rows, prepared, legacy_scopes = {}, {}, {}
    for key, old in prior_plan['runs'].items():
        original = Path(old['new_run'])
        rt = original / 'runtime'
        identity = read(rt / 'driver-identity.json')
        assert helper.same(identity) and identity['uid'] == 20001 and identity['namespace'] == old['namespace']
        check_return_invocation(identity, previous_stage, previous_module, key)
        identity.update(cmdline_sha256=helper.proc(identity['pid'])['cmdline_sha256'], match_cmdline=True)
        assert helper.same(identity), 'Returned driver identity changed during preparation'
        assert not (rt / 'finished.json').exists(), 'Finished RLT requires a recovery audit'
        cfg = helper.config(rt / 'resolved.yaml')
        tb = original / 'tensorboard/config.yaml'
        if not tb.is_file():
            tb = original / original.name / 'tensorboard/config.yaml'
        # Preempting WM-priority jobs does not wait for their first training round.
        if tb.is_file():
            assert cfg == helper.config(tb)
        gpu = old['gpus'][0]
        assert cfg['cluster']['component_placement'] == {'actor,env,rollout': gpu}
        recovery = helper.select_recovery(original, cfg, prior_plan['repo'])
        assert recovery['mode'] == 'resume_checkpoint' and recovery['checkpoint']
        actors = helper.active(live, old['namespace'])
        jobs = {a['job_id'] for a in actors}
        assert len(jobs) <= 1
        helper.validate_actor_rows(actors, old['namespace'], jobs)
        tree = helper.process_tree({identity['pid']} | {a['pid'] for a in actors if a.get('pid')})
        assert {r['pid'] for r in helper.gpu_processes([gpu])} <= set(tree)
        new_run = original.with_name(original.name + '-after-' + stage.name)
        namespace = 'rlt-opendw-return-' + stage.name[-35:] + '-g' + str(gpu)
        assert not new_run.exists() and not helper.active(live, namespace)
        newcfg, changes = helper.resumed_config(cfg, original, new_run, recovery['checkpoint']['path'])
        env = read(rt / 'environment.json')
        assert not any(k in env for k in helper.MASKS)
        env = {k: v.replace(str(original), str(new_run)).replace(old['namespace'], namespace) for k, v in env.items()}
        legacy_path = env.get('RLINF_OPENDW_GPU_SCOPE_MANIFEST')
        if legacy_path:
            legacy = read(owned(legacy_path))
            assert legacy['uid'] == 20001
            legacy_scopes[legacy_path] = {'manifest_sha256': sha(legacy_path),
                'bootstrap_dir': str(Path(legacy['bootstrap_path']).parent), 'marker_path': legacy['marker_path']}
        rows[key] = dict(gpus=[gpu], original_run=str(original), original_namespace=old['namespace'],
            original_identity=identity, original_jobs=sorted(jobs), new_run=str(new_run), namespace=namespace,
            entry=old['entry'], recovery=recovery, config_changes=changes,
            dependencies=helper.dependency_snapshot(cfg), latest_metrics_before=helper.latest_metrics(original),
            original_runtime_sha256={n: sha(rt / n) for n in ('resolved.yaml', 'environment.json', 'driver-identity.json')})
        prepared[key] = cfg, newcfg, env
    stage.mkdir(mode=0o700, parents=True, exist_ok=True)
    frozen = {}
    sources = {'rlt_checkpoint_lifecycle.py': prior_plan['helper_path'],
               'next_six_ops.py': prior_plan['next_six_ops'], 'base_rlt_gpu567_cycle.py': base_module,
               SCRIPT_NAME: Path(__file__)}
    for name, path in sources.items():
        dest = stage / name
        assert not dest.exists() and dest.resolve() != Path(path).resolve()
        with dest.open('xb') as f:
            f.write(Path(path).read_bytes())
        dest.chmod(0o500)
        frozen[str(dest)] = sha(dest)
    for key, (cfg, newcfg, env) in prepared.items():
        pre = stage / 'prepared' / key
        pre.mkdir(parents=True, mode=0o700)
        OmegaConf.save(OmegaConf.create(cfg), pre / 'original.yaml', resolve=True)
        OmegaConf.save(OmegaConf.create(newcfg), pre / 'resolved.yaml', resolve=True)
        helper.save(pre / 'environment.json', env)
        rows[key]['prepared_sha256'] = {n: sha(pre / n) for n in ('original.yaml', 'resolved.yaml', 'environment.json')}
    frozen.update({str(previous_stage / 'plan.json'): sha(previous_stage / 'plan.json'),
                   str(previous_module): sha(previous_module)})
    for key in ('owner_plan', 'final', 'identity', 'cleanup', 'release', 'owner_return',
                'return_started', 'combined_plan', 'combined_return', 'child_return'):
        frozen[evidence[key]] = evidence[key + '_sha256']
    frozen.update({path: row['manifest_sha256'] for path,row in legacy_scopes.items()})
    p = {k: prior_plan[k] for k in ('repo', 'head', 'uid', 'python', 'ray_address', 'ray_dashboard_url')}
    p.update(group=group, cycle_id=stage.name, time=helper.now(), script_sha256=sha(__file__),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        base_module=str(stage / 'base_rlt_gpu567_cycle.py'), helper_path=str(stage / 'rlt_checkpoint_lifecycle.py'),
        next_six_ops=str(stage / 'next_six_ops.py'), old_owner=old_owner, frozen_files=frozen,
        source_sha256=copy.deepcopy(prior_plan['source_sha256']), runs=rows,
        management_namespace='opendw-returned-' + stage.name[-30:],
        previous_cycle=str(previous_stage), completed_owner=str(parent_owner),
        scope_activation=str(scope_activation) if scope_activation else None,
        scope_manifest=str(scope_manifest) if scope_manifest else None, scope_id=scope_id,
        legacy_scopes=legacy_scopes)
    assert bool(scope_activation) == bool(scope_manifest) == bool(scope_id)
    helper.save(stage / 'plan.json', p)
    install_helper(stage)
    load_plan(stage)
    B.watch_targets(read(B.WATCH), rows)
    helper.save(stage / 'prepared.json', {'time': helper.now(), 'previous_cycle': str(previous_stage),
        'checkpoint_steps': {k: r['recovery']['checkpoint']['step'] for k, r in rows.items()},
        'preempt_before_first_round_allowed': True})
    return p


def prepare_adopted(stage, previous_stage, previous_module, parent_owner, base_module,
                    scope_activation=None, scope_manifest=None, scope_id=None, handoff_intent=None):
    """Transfer an already stopped RLT checkpoint contract without restarting RLT."""
    assert handoff_intent and bool(scope_activation) == bool(scope_manifest) == bool(scope_id)
    stage, previous_stage = owned(stage, False), owned(previous_stage)
    assert not stage.exists(), 'Adoption must use a fresh cycle directory'
    previous_module, base_module = owned(previous_module), owned(base_module)
    assert sha(base_module) == BASE_SHA
    adopted_parent(parent_owner, previous_stage, previous_module, handoff_intent)
    prior = load('adopted_prior_' + stage.name, previous_module)
    prior.install_helper(previous_stage)
    old = prior.load_plan(previous_stage)
    helper = prior.H
    group, identity, frozen = adopted_parent(parent_owner, previous_stage, previous_module, handoff_intent, helper)
    assert old['head'] == HEADS[group]
    stopped = read(previous_stage / 'rlt-stopped.json')
    assert set(stopped['runs']) == set(old['runs'])
    live = helper.actors(old)
    rows, environments, legacy_scopes = copy.deepcopy(old['runs']), {}, {}
    for key, row in rows.items():
        assert not helper.same(row['original_identity'])
        assert not helper.active(live, row['original_namespace']) and not helper.active(live, row['namespace'])
        assert not Path(row['new_run']).exists(), 'The previous RLT return destination was already used'
        pre = previous_stage / 'prepared' / key
        for name in ('original.yaml', 'environment.json'):
            assert sha(pre / name) == row['prepared_sha256'][name]
        state = stopped['runs'][key]
        assert sha(pre / 'resolved.yaml') == state['resolved_sha256']
        recovery = state['recovery']
        assert recovery['mode'] == 'resume_checkpoint' and recovery['checkpoint']
        cp = recovery['checkpoint']
        cfg = helper.config(pre / 'resolved.yaml')
        assert cfg['runner']['resume_dir'] == cp['path']
        assert cfg['cluster']['component_placement'] == {'actor,env,rollout': row['gpus'][0]}
        checked = helper.inspect_checkpoint(cp['path'], cfg, old['repo'])
        assert checked['step'] == cp['step'] and checked['contract_sha256'] == cp['contract_sha256']
        for relative, digest in cp['small_sha256'].items():
            assert sha(Path(cp['path']) / 'actor' / relative) == digest
        assert helper.dependency_snapshot(cfg) == row['dependencies']
        env = read(pre / 'environment.json')
        assert not any(k in env for k in helper.MASKS)
        legacy_path = env.get('RLINF_OPENDW_GPU_SCOPE_MANIFEST')
        if legacy_path:
            legacy = read(owned(legacy_path))
            assert legacy['uid'] == 20001
            legacy_scopes[legacy_path] = {'manifest_sha256': sha(legacy_path),
                'bootstrap_dir': str(Path(legacy['bootstrap_path']).parent), 'marker_path': legacy['marker_path']}
        environments[key] = env
        runtime = Path(row['original_run']) / 'runtime'
        row['original_runtime_sha256'] = {name: sha(runtime / name)
            for name in ('resolved.yaml', 'environment.json', 'driver-identity.json')}
        row['recovery'], row['config_changes'] = copy.deepcopy(recovery), copy.deepcopy(state['config_changes'])
        row['prepared_sha256'] = {name: sha(pre / name)
            for name in ('original.yaml', 'resolved.yaml', 'environment.json')}
    # Every live/source/CP check above finishes before creating the new cycle.
    stage.mkdir(mode=0o700, parents=True)
    sources = {'rlt_checkpoint_lifecycle.py': old['helper_path'], 'next_six_ops.py': old['next_six_ops'],
               'base_rlt_gpu567_cycle.py': base_module, SCRIPT_NAME: Path(__file__)}
    for name, path in sources.items():
        dest = stage / name
        with dest.open('xb') as stream:
            stream.write(Path(path).read_bytes())
        dest.chmod(0o500)
        frozen[str(dest)] = sha(dest)
    for key in rows:
        dest = stage / 'prepared' / key
        dest.mkdir(parents=True, mode=0o700)
        for name in ('original.yaml', 'resolved.yaml', 'environment.json'):
            source, target = previous_stage / 'prepared' / key / name, dest / name
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
            target.chmod(0o400)
            frozen[str(target)] = sha(target)
    frozen.update({str(previous_stage / 'plan.json'): sha(previous_stage / 'plan.json'),
                   str(previous_module): sha(previous_module)})
    frozen.update({path: row['manifest_sha256'] for path, row in legacy_scopes.items()})
    p = {key: old[key] for key in ('repo', 'head', 'uid', 'python', 'ray_address', 'ray_dashboard_url')}
    p.update(group=group, cycle_id=stage.name, time=helper.now(), script_sha256=sha(__file__),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        base_module=str(stage / 'base_rlt_gpu567_cycle.py'), helper_path=str(stage / 'rlt_checkpoint_lifecycle.py'),
        next_six_ops=str(stage / 'next_six_ops.py'), old_owner=identity, frozen_files=frozen,
        source_sha256=copy.deepcopy(old['source_sha256']), runs=rows,
        management_namespace='opendw-adopted-' + stage.name[-30:],
        previous_cycle=str(previous_stage), completed_owner=str(parent_owner),
        scope_activation=str(scope_activation) if scope_activation else None,
        scope_manifest=str(scope_manifest) if scope_manifest else None, scope_id=scope_id,
        legacy_scopes=legacy_scopes, adopted_from={'cycle': str(previous_stage), 'owner': str(parent_owner),
            'handoff_intent': str(handoff_intent), 'handoff_intent_sha256': sha(handoff_intent),
            'stopped_sha256': sha(previous_stage / 'rlt-stopped.json')})
    helper.save(stage / 'plan.json', p)
    adopted_stop = copy.deepcopy(stopped)
    adopted_stop.update(time=helper.now(), cycle_id=stage.name, adopted_from=p['adopted_from'])
    helper.save(stage / 'rlt-stopped.json', adopted_stop)
    helper.save(stage / 'monitor-retired.json', {'time': helper.now(), 'identity': identity,
        'monitor_only': True, 'reason': 'Previous OpenDW owner ended in the explicit locked handoff; no signal sent'})
    helper.save(stage / 'adopted.json', {'time': helper.now(), 'cycle_id': stage.name,
        'physical_gpus': [4] if group == 'gpu4' else [5, 6, 7], 'adopted_from': p['adopted_from'],
        'checkpoint_steps': {key: value['recovery']['checkpoint']['step'] for key, value in stopped['runs'].items()},
        'rlt_remained_stopped': True})
    install_helper(stage)
    load_plan(stage)
    B.watch_targets(read(B.WATCH), rows)
    return p


def stop(stage):
    return B.stop(stage)


def finalize_stopped(stage):
    return B.finalize_stopped(stage)


def status(stage):
    return B.status(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle-dir', type=Path, required=True)
    parser.add_argument('action', choices=('driver', 'stop', 'resume', 'status', 'finalize-stopped'))
    parser.add_argument('--key')
    parser.add_argument('--release-receipt')
    args = parser.parse_args()
    install_helper(args.cycle_dir)
    if args.action == 'driver':
        H.driver(args.cycle_dir, args.key)
        return
    if args.action == 'status':
        print(json.dumps(status(args.cycle_dir)))
        return
    with (args.cycle_dir / 'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'stop':
            result = stop(args.cycle_dir)
        elif args.action == 'finalize-stopped':
            result = finalize_stopped(args.cycle_dir)
        else:
            result = H.resume(args.cycle_dir, args.release_receipt)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
