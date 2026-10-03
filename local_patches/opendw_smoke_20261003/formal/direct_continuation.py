"""Immediate, identity-scoped WM smoke -> formal handoff; no RLT in between.

Hold the old return lock, request the old owner's normal cleanup, adopt its
already-held RLT checkpoints, and launch the formal owner. No CP/nonzero-gradient
gate. Before transfer succeeds, every failure stays under this supervisor's
cleanup/return responsibility. Use a fresh directory and a reviewed ready.json.
"""
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


S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
OLD_OWNER = S / 'runs/multigpu-v1'
NEW_OWNER = S / 'runs/formal-v1'
REPO = S / 'rlinf-multigpu-v1'
OLD_IDENTITY = dict(pid=3230235, uid=20001, start=701253070)
BASE_OWNER = S / 'multigpu-control-v1/multigpu/opendw_multigpu_owner.py'
BASE_567 = S / 'multigpu-control-v1/multigpu/rlt_gpu567_cycle.py'
CONTROL_REF = '2151a08ee1bd75df1bef0d8190e594bd5c7f7977'
CONTROL_CONFIG = 'examples/embodiment/config/sz2_can256_clean_resume_n32-20260927-v1.yaml'
NATIVE_ASSETS = Path('/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support')
REQUIRED = ('direct_continuation.py', 'opendw_formal_owner.py', 'build_formal_config.py',
            'rlt_returned_cycle.py', 'rlt_returned_multigpu_cycle.py', 'post_borrow_hook.py',
            'graphics_scope_prepare.py', 'graphics_scope_runtime.py', 'graphics_scope_bootstrap.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, data):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_ready(path, code):
    ready = read(path)
    assert ready['schema'] == 1 and ready['all_cpu_tests_passed'] is True
    pinned = ready['source_sha256']
    for name in REQUIRED:
        source = code / name
        assert pinned.get(str(source)) == sha(source), 'Unready source: ' + str(source)
    for source, expected in pinned.items():
        assert sha(source) == expected, 'Ready source changed: ' + source
    return ready


def never_returned(plan):
    cycle = Path(plan['lifecycle_path'])
    combined = read(cycle / 'plan.json')
    for directory in [cycle] + [Path(row['path']) for row in combined['children'].values()]:
        for name in ('return-started.json', 'resumed-dispatched.json'):
            assert not (directory / name).exists(), 'RLT return already began: ' + str(directory / name)
        assert not list(directory.glob('gpu*-launched.json')), 'RLT was already launched'
    return combined


def signal_exact(M, identity, sig):
    assert M.H.same(identity), 'Target identity is no longer live'
    fd = M.pidfd_open(identity['pid'])
    try:
        assert M.H.same(identity), 'Target identity changed while opening pidfd'
        M.pidfd_send(fd, sig)
    finally:
        os.close(fd)


def clean_environment(env):
    result = copy.deepcopy(env)
    assert not any(key in result for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
    assert not result.get('RLINF_OPENDW_GPU_SCOPE_MANIFEST')
    assert not result.get('RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST')
    assert not result.get('LD_PRELOAD')
    for key in ('__GL_APPLICATION_PROFILE', '__GL_APPLICATION_PROFILE_LOG'):
        result.pop(key, None)
    return result


def plan_inputs(code, state, old, ready):
    """Only new local configuration/staged scope files; old WM keeps running."""
    builder = load('direct_formal_builder', code / 'build_formal_config.py')
    graphics = load('direct_graphics_preparer', code / 'graphics_scope_prepare.py')
    full = next(row for row in old['trials'] if row['key'] == 'n64_full')
    full_cfg = read(full['config'])
    assert sha(full['config']) == full['config_sha256']
    raw_control = subprocess.check_output(['git', '-C', str(REPO), 'show', CONTROL_REF + ':' + CONTROL_CONFIG], timeout=30)
    control = json.loads(raw_control)
    record(state / 'control-config.json', control)
    seed_source = REPO / 'rlinf/envs/robotwin/seeds/eval_seeds.json'
    seed_record = read(seed_source)['adjust_bottle']
    values = seed_record['success_seeds']
    assert len(values) == len(set(values)) == 150 and all(type(value) is int for value in values)
    seeds = state / 'adjust-bottle-native-eval-seeds.json'
    record(seeds, {'adjust_bottle': seed_record})
    record(state / 'native-seed-provenance.json', dict(source=str(seed_source), sha256=sha(seed_source),
           selected_task='adjust_bottle', count=150, destination=str(seeds), selection='existing standard seed list; no generation/filter'))
    assert NATIVE_ASSETS.is_dir()
    services = copy.deepcopy(old['services'])
    for row in services:
        argv = row['argv']
        assert argv.count('--output-dir') == 1
        argv[argv.index('--output-dir') + 1] = str(NEW_OWNER / 'services' / row['key'] / 'records')
    config, contract = builder.build(full_cfg, control, name='opendw-adjust-bottle-formal-v1',
            run_dir=str(NEW_OWNER / 'formal/run'), services=[row['url'] for row in services],
            native_assets=str(NATIVE_ASSETS), native_eval_seeds=str(seeds))
    assert config['runner'].get('resume_dir') is None and config['runner'].get('ckpt_path') is None
    config_path = state / 'opendw-adjust-bottle-formal-v1.yaml'
    record(config_path, config)
    record(state / 'formal-contract.json', contract)
    base_env = clean_environment(read(old['environment_file']))
    env_path = state / 'environment.json'
    record(env_path, base_env)
    env_path.chmod(0o600)
    prior_cycle = read(Path(old['lifecycle_path']) / 'plan.json')
    prior4 = read(Path(prior_cycle['children']['gpu4']['path']) / 'plan.json')
    # Read the already-frozen GPU4 scope through its recorded manifest path.
    legacy_path = prior4.get('graphics_scope', {}).get('manifest_path')
    legacy_paths = [legacy_path] if legacy_path else []
    if not legacy_paths:
        for row in prior4['runs'].values():
            pre = Path(prior_cycle['children']['gpu4']['path']) / 'prepared' / 'gpu4' / 'environment.json'
            if pre.is_file():
                value = read(pre).get('RLINF_OPENDW_GPU_SCOPE_MANIFEST')
                if value:
                    legacy_paths.append(value)
    assert len(set(legacy_paths)) == 1, 'Need the frozen legacy scope manifest for profile audit'
    legacy = read(legacy_paths[0])
    assert sha(legacy['profile_path']) == legacy['profile_sha256']
    scope_dir = (state / 'graphics-scope').resolve()
    scope_id = 'opendwformal20261004v1'
    staged = graphics.prepare_stage(scope_dir, scope_id, [legacy['profile_path']], inherited=base_env)
    assert staged['global_profile_installed'] is False
    return dict(full=full, config=config_path, services=services, environment=env_path, seeds=seeds,
                scope_dir=scope_dir, scope_id=scope_id, staged=staged)


def certify_release(M, old, state):
    assert not M.H.same(OLD_IDENTITY), 'Previous WM owner is still live'
    final, cleanup, release = [read(OLD_OWNER / name) for name in ('final.json', 'cleanup.json', 'smoke-release.json')]
    assert cleanup['all_stopped'] is True and release['all_workers_stopped'] is True
    assert release['cycle_id'] == Path(old['lifecycle_path']).name and release['gpus'] == [4, 5, 6, 7]
    assert final['rlt_return_dispatched'] is False and final['rlt_borrowed'] is True
    assert final['recovery_error']['type'] == 'BlockingIOError', 'Old return was not intercepted by the explicit lock'
    assert all(not M.H.same(row) for row in release['managed_processes'])
    catalog = read(OLD_OWNER / 'process-catalog.json')['processes']
    assert all(not M.H.same(row) for row in catalog)
    contexts = M.H.gpu_processes(list(range(8)))
    own = [row for row in contexts if (M.H.proc(row['pid']) or {}).get('uid') == 20001]
    assert not own, 'Owned compute/graphics contexts remain after old cleanup'
    never_returned(old)
    record(state / 'old-release-verified.json', dict(time=M.H.now(), previous_owner=OLD_IDENTITY,
            final_sha256=sha(OLD_OWNER / 'final.json'), cleanup_sha256=sha(OLD_OWNER / 'cleanup.json'),
            release_sha256=sha(OLD_OWNER / 'smoke-release.json'), owned_contexts_all8=own))
    return release


def make_formal_plan(code, state, old, inputs, ready, combined, adoption_receipt):
    source = dict(old['source_sha256'])
    source.update(ready['source_sha256'])
    combined_plan = read(combined / 'plan.json')
    source[str(combined / 'rlt_returned_multigpu_cycle.py')] = sha(combined / 'rlt_returned_multigpu_cycle.py')
    for child in combined_plan['children'].values():
        source[child['module']] = sha(child['module'])
        child_plan = read(Path(child['path']) / 'plan.json')
        source.update(child_plan.get('frozen_files', {}))
    # Artifact hashes are checked when used; source pins never include changing logs.
    staged = inputs['staged']
    plan = dict(mode='multigpu_formal', start_mode='direct_start_user_override_20261004',
                owner_dir=str(NEW_OWNER), lifecycle_path=str(combined),
                lifecycle_module=str(combined / 'rlt_returned_multigpu_cycle.py'),
                base_owner_module=str(BASE_OWNER), python=old['python'], repo=str(REPO), repo_head=old['repo_head'],
                physical_gpus=[4, 5, 6, 7], environment_file=str(inputs['environment']), source_sha256=source,
                services=inputs['services'], trials=[dict(key='formal', num_envs=64, episode_steps=384,
                  config=str(inputs['config']), config_sha256=sha(inputs['config']),
                  namespace='opendw_sz3_formal_v1', timeout_seconds=60 * 86400)], restore_wait_seconds=60,
                native_eval_seeds_sha256=sha(inputs['seeds']),
                protocol_reference=dict(config=inputs['full']['config'], sha256=inputs['full']['config_sha256']),
                borrow_adoption=dict(receipt=str(adoption_receipt), sha256=sha(adoption_receipt)),
                handoff_evidence=dict(owner_final_path=str(OLD_OWNER / 'final.json'), owner_final_sha256=sha(OLD_OWNER / 'final.json')),
                graphics_scope=dict(scope_dir=str(inputs['scope_dir']),
                  environment_fragment_file=staged['environment_fragment_file'],
                  environment_fragment_sha256=staged['environment_fragment_sha256'],
                  post_borrow_hook_module=str(code / 'post_borrow_hook.py'), prepare_module=str(code / 'graphics_scope_prepare.py'),
                  activation_receipt=str(combined / 'scope-activation.json')),
                budget=dict(formal_training=True, effective_runner_iterations=200, N=64, G=8, R=8,
                  C=32, L=384, scheduled_optimizer_steps_per_iteration=6, save_interval=10, native_eval_interval=10,
                  user_override='Direct formal start; no completed-smoke, CP1 or nonzero-gradient gate'))
    target = state / 'formal-plan.json'
    record(target, plan)
    return target, plan


def emergency_cleanup(M, plan, state, identity, label):
    """Only this exact owner's namespace/job/token, using reviewed cleanup."""
    if identity and M.H.same(identity):
        signal_exact(M, identity, signal.SIGKILL)
        deadline = time.monotonic() + 15
        while M.H.same(identity) and time.monotonic() < deadline:
            time.sleep(1)
    assert not identity or not M.H.same(identity), 'Owner remains live; no competing return'
    catalog = M.Catalog(state / ('emergency-' + label + '-process-catalog.json'), plan['token'])
    prior = Path(plan['owner_dir']) / 'process-catalog.json'
    if prior.exists():
        for row in read(prior)['processes']:
            catalog.add(row, row['phase'], 'Pinned previous owner catalog')
    result = M.cleanup(plan, catalog)
    record(state / ('emergency-' + label + '-cleanup.json'), result)
    assert result['all_stopped']
    return list(catalog.rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready-manifest', type=Path, required=True)
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    code = Path(__file__).absolute().parent
    ready = validate_ready(args.ready_manifest, code)
    sys.path.insert(0, str(code))
    state = code.parent / 'direct-state'
    state.mkdir(mode=0o700, exist_ok=False)
    own_lock = (state / 'owner.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read(OLD_OWNER / 'owner-plan.json')
    assert old['owner_dir'] == str(OLD_OWNER) and old['repo'] == str(REPO)
    assert old['owner_script_sha256'] == sha(BASE_OWNER)
    M = load('direct_old_owner', BASE_OWNER)
    M.load_lifecycle(old)
    M.validate(old, frozen=True)
    me = M.H.proc(os.getpid())
    record(state / 'owner-identity.json', me)
    record(state / 'ready-used.json', dict(path=str(args.ready_manifest), sha256=sha(args.ready_manifest)))
    held, signaled, transferred, combined_module, combined_path = None, False, False, None, None
    failure, recovery = None, None
    formal_child = None
    try:
        previous = read(OLD_OWNER / 'owner-identity.json')
        assert all(previous[key] == value for key, value in OLD_IDENTITY.items()) and M.H.same(previous)
        assert not (OLD_OWNER / 'final.json').exists() and not NEW_OWNER.exists()
        R = load('direct_returned_cycle', code / 'rlt_returned_cycle.py')
        C = load('direct_returned_combined', code / 'rlt_returned_multigpu_cycle.py')
        assert callable(R.prepare_adopted) and callable(C.prepare_adopted) and callable(C.verify_adoption)
        inputs = plan_inputs(code, state, old, ready)
        previous_cycle = never_returned(old)
        lock_path = Path(old['lifecycle_path']) / 'operation.lock'
        assert lock_path.is_file()
        held = lock_path.open('a')
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        never_returned(old)
        intent = dict(kind='opendw-formal-adopt-held-rlt', parent_owner=str(OLD_OWNER), lifecycle_path=old['lifecycle_path'],
                      owner_identity=previous, holder_identity=me, physical_gpus=[4, 5, 6, 7], time=M.H.now(),
                      owner_plan_sha256=sha(OLD_OWNER / 'owner-plan.json'),
                      combined_plan_sha256=sha(Path(old['lifecycle_path']) / 'plan.json'),
                      child_plan_sha256={key: row['plan_sha256'] for key, row in previous_cycle['children'].items()},
                      child_module_sha256={key: row['module_sha256'] for key, row in previous_cycle['children'].items()})
        intent_path = state / 'handoff-intent.json'
        record(intent_path, intent)
        signal_exact(M, previous, signal.SIGTERM)
        signaled = True
        record(state / 'handoff-signal.json', dict(time=M.H.now(), identity=previous, signal=int(signal.SIGTERM)))
        deadline = time.monotonic() + 900
        while M.H.same(previous) or not (OLD_OWNER / 'final.json').is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError('Old owner cleanup/terminal handoff exceeded 900 seconds')
            M.atomic(state / 'status.json', dict(time=M.H.now(), phase='waiting_exact_old_cleanup', previous_owner=previous))
            time.sleep(2)
        certify_release(M, old, state)
        combined_path = state / 'adopted-cycle'
        children = {}
        for key in ('gpu4', 'gpu567'):
            prior = previous_cycle['children'][key]
            child = state / ('adopted-' + key)
            R.prepare_adopted(child, Path(prior['path']), Path(prior['module']), OLD_OWNER, BASE_567,
               scope_activation=combined_path / 'scope-activation.json', scope_manifest=inputs['scope_dir'] / 'scope.json',
               scope_id=inputs['scope_id'], handoff_intent=intent_path)
            children[key] = dict(path=str(child), module=str(child / 'rlt_returned_cycle.py'))
        C.prepare_adopted(combined_path, children, handoff_intent=intent_path)
        combined_module = load('direct_adopted_combined_live', combined_path / 'rlt_returned_multigpu_cycle.py')
        combined_module.install_helper(combined_path)
        adoption = combined_path / 'adopted.json'
        combined_module.verify_adoption(combined_path, adoption)
        plan_path, plan = make_formal_plan(code, state, old, inputs, ready, combined_path, adoption)
        owner = load('direct_formal_preflight', code / 'opendw_formal_owner.py')
        module = owner.install(plan)
        module.validate(plan)
        # The new owner accepts the already-held cycle from startup, including
        # failure while loading CPU services. Release old lock only after this
        # new responsibility and its exact startup identity are recorded.
        argv = [plan['python'], '-u', '-B', str(code / 'opendw_formal_owner.py'), '--plan', str(plan_path), 'owner']
        environment = read(plan['environment_file'])
        with (state / 'formal-owner.log').open('x') as log:
            formal_child = subprocess.Popen(argv, cwd=plan['repo'], env=environment,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = M.H.proc(formal_child.pid)
        assert identity and identity['uid'] == 20001
        record(state / 'formal-owner-launched.json', dict(time=M.H.now(), identity=identity, argv=argv,
               plan=str(plan_path), plan_sha256=sha(plan_path), borrow_adoption=str(adoption), borrow_adoption_sha256=sha(adoption)))
        deadline = time.monotonic() + 120
        while not (NEW_OWNER / 'owner-identity.json').exists():
            assert formal_child.poll() is None, 'Formal owner exited before accepting responsibility'
            if time.monotonic() >= deadline:
                raise TimeoutError('Formal owner startup receipt deadline')
            time.sleep(1)
        accepted = read(NEW_OWNER / 'owner-identity.json')
        assert all(accepted[key] == identity[key] for key in ('pid', 'uid', 'start'))
        assert M.H.same(accepted) and read(NEW_OWNER / 'owner-plan.json')['borrow_adoption'] == plan['borrow_adoption']
        transferred = True
        record(state / 'handoff-completed.json', dict(time=M.H.now(), status='formal_owner_running',
               formal_owner=accepted, plan=str(plan_path), old_rlt_was_not_restarted=True,
               no_learning_or_checkpoint_gate=True, responsible_owner=str(NEW_OWNER)))
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        record(state / 'failure.json', failure)
    finally:
        if not transferred and signaled:
            try:
                if formal_child is not None and formal_child.poll() is None:
                    ident = M.H.proc(formal_child.pid)
                    signal_exact(M, ident, signal.SIGTERM)
                    try:
                        formal_child.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        signal_exact(M, ident, signal.SIGKILL)
                        formal_child.wait(timeout=15)
                if (NEW_OWNER / 'final.json').is_file() and read(NEW_OWNER / 'final.json').get('rlt_return_dispatched'):
                    recovery = dict(status='formal_owner_returned_rlt', receipt=str(NEW_OWNER / 'final.json'))
                else:
                    managed = emergency_cleanup(M, old, state, OLD_IDENTITY, 'old')
                    if (NEW_OWNER / 'owner-plan.json').is_file():
                        newplan = read(NEW_OWNER / 'owner-plan.json')
                        newidentity = read(NEW_OWNER / 'owner-identity.json') if (NEW_OWNER / 'owner-identity.json').is_file() else None
                        managed.extend(emergency_cleanup(M, newplan, state, newidentity, 'formal'))
                    assert not M.H.gpu_processes([4, 5, 6, 7]), 'WM contexts remain; return stays blocked'
                    if combined_module is None:
                        M.load_lifecycle(old)
                    target, helper = ((combined_path, combined_module.H) if combined_module is not None
                                      else (Path(old['lifecycle_path']), M.C.H))
                    release = state / 'supervisor-wm-release.json'
                    record(release, dict(time=M.H.now(), cycle_id=target.name, gpus=[4, 5, 6, 7],
                       all_workers_stopped=True, managed_processes=managed, terminal_status='failed', reason='handoff_failed'))
                    if held is not None:
                        fcntl.flock(held, fcntl.LOCK_UN)
                        held.close()
                        held = None
                    recovery = dict(status='rlt_return_dispatched', result=helper.resume(target, release))
                record(state / 'recovery.json', recovery)
            except BaseException as exc:
                recovery = dict(status='recovery_failed_requires_attention', type=type(exc).__name__,
                                error=str(exc), traceback=traceback.format_exc())
                record(state / 'recovery-failed.json', recovery)
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
            held.close()
        record(state / 'final.json', dict(time=M.H.now(), status='formal_running' if transferred else 'not_transferred',
               signaled_old_owner=signaled, transferred=transferred, failure=failure, recovery=recovery,
               original_rlt_untouched_by_this_supervisor=not signaled))
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
