"""One exact formal-v2 -> batch-16 handoff, preserving held RLT and policy CP.

CPU readiness and the complete external checkpoint must validate before any
signal. No old source, owner plan, or RLT borrowing transaction is replayed.
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
D = S / 'formal-b16-control-v1'
OLD_OWNER = S / 'runs/formal-v2'
NEW_OWNER = S / 'runs/formal-b16-v1'
OLD_IDENTITY = dict(pid=3242048, uid=20001, start=706364464)
OLD_WRAPPER = S / 'formal-control-v2/formal/opendw_formal_owner.py'
CODE = D / 'code'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ready_manifest(path):
    value = read(path)
    assert value['schema'] == 1 and value['all_cpu_tests_passed'] is True
    assert value['wm_batch_size'] == 16
    for name in ('batch16_handoff.py', 'batch16_formal_owner.py', 'opendw_service_batched.py', 'wm_batch.py',
                 'opendw_action_telemetry.py', 'opendw_reward.py'):
        target = CODE / name
        assert value['source_sha256'].get(str(target)) == sha(target), 'Unready batch-16 source: ' + name
    for name, digest in value['source_sha256'].items():
        assert sha(name) == digest, 'Ready source changed: ' + name
    return value


def make_inputs(old, ready, resume_path, state, template_path=None):
    F = load('batch16_checkpoint_preflight', CODE / 'batch16_formal_owner.py')
    resumed = F.verify_resume(resume_path)
    old_formal = next(row for row in old['trials'] if row['key'] == 'formal')
    assert sha(old_formal['config']) == old_formal['config_sha256']
    cfg = read(old_formal['config'])
    assert cfg['runner'].get('resume_dir') is None and cfg['runner'].get('ckpt_path') is None
    old_name = cfg['runner']['logger']['experiment_name']
    new_name = 'opendw-adjust-bottle-formal-b16-v1'
    formal = json.loads(json.dumps(cfg).replace(str(OLD_OWNER / 'formal'), str(NEW_OWNER / 'formal')).replace(old_name, new_name))
    formal['runner']['resume_dir'] = resumed['checkpoint_path']
    smoke = copy.deepcopy(formal)
    smoke['runner'].update(max_epochs=1, max_steps=1, save_interval=1, val_check_interval=1, resume_dir=None, ckpt_path=None)
    smoke['env']['train'].update(rollout_epoch=1, max_episode_steps=32, max_steps_per_rollout_epoch=32)
    smoke['env']['eval'].update(max_episode_steps=32, max_steps_per_rollout_epoch=32)
    smoke['env']['eval']['task_config']['step_lim'] = 32
    smoke['actor']['global_batch_size'] = 64
    smoke = json.loads(json.dumps(smoke).replace(str(NEW_OWNER / 'formal'), str(NEW_OWNER / 'startup_smoke'))
                       .replace(new_name, 'opendw-adjust-bottle-startup-b16-v1'))
    trials = []
    for key, value, length, timeout in [('startup_smoke', smoke, 32, 3600), ('formal', formal, 384, 60 * 86400)]:
        path = state / (key + '.yaml')
        record(path, value)
        trials.append(dict(key=key, num_envs=64, episode_steps=length, config=str(path), config_sha256=sha(path),
            namespace='opendw_sz3_' + key + '_b16_v1', timeout_seconds=timeout))
    services = copy.deepcopy(old['services'])
    for service in services:
        argv = service['argv']
        scripts = [i for i, arg in enumerate(argv) if Path(arg).name == 'opendw_service.py']
        assert len(scripts) == 1 and '--wm-batch-size' not in argv
        argv[scripts[0]] = str(CODE / 'opendw_service_batched.py')
        assert argv.count('--output-dir') == 1
        argv[argv.index('--output-dir') + 1] = str(NEW_OWNER / 'services' / service['key'] / 'records')
        argv += ['--execution-mode', 'batched', '--wm-batch-size', '16']
    environment = read(old['environment_file'])
    assert not any(k in environment for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
        'RLINF_OPENDW_GPU_SCOPE_MANIFEST', 'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST', 'LD_PRELOAD'))
    env_path = state / 'environment.json'
    record(env_path, environment)
    env_path.chmod(0o600)
    scope = copy.deepcopy(old['graphics_scope'])
    scope.pop('audited_unreadable_cpu_processes', None)
    active = read(scope['activation_receipt'])
    scope['reuse_active'] = dict(source_owner=str(OLD_OWNER), source_owner_plan_sha256=sha(OLD_OWNER / 'owner-plan.json'),
        activation_sha256=sha(scope['activation_receipt']), manifest_sha256=active['manifest_sha256'])
    source = dict(old['source_sha256'])
    source.update(ready['source_sha256'])
    source[str(OLD_WRAPPER)] = sha(OLD_WRAPPER)
    source[active['runtime_path']] = sha(active['runtime_path'])
    plan = dict(mode='multigpu_formal', start_mode='direct_start_user_override_20261004', startup_smoke=True,
        wm_batch_size=16, owner_dir=str(NEW_OWNER), lifecycle_path=old['lifecycle_path'],
        lifecycle_module=old['lifecycle_module'], base_owner_module=old['base_owner_module'],
        base_formal_wrapper=str(OLD_WRAPPER), python=old['python'], repo=old['repo'], repo_head=old['repo_head'],
        physical_gpus=[4, 5, 6, 7], environment_file=str(env_path), source_sha256=source, services=services,
        trials=trials, restore_wait_seconds=60, native_eval_seeds_sha256=old['native_eval_seeds_sha256'],
        protocol_reference=old['protocol_reference'], graphics_scope=scope,
        resume_checkpoint=dict(receipt=str(resume_path), sha256=sha(resume_path)),
        budget=dict(effective_runner_iterations=200, resumed_completed_iterations=resumed['completed_step'],
            remaining_iterations=200-resumed['completed_step'], N=64, G=8, R=8, C=32, L=384, wm_batch_size=16,
            actor_global_batch=2048, actor_microbatch=8, update_epoch=2, save_interval=10, native_eval_interval=10,
            startup_smoke=dict(N=64, G=8, R=1, L=32, global_batch=64, native_eval_N=32, native_eval_L=32,
                initialization='original SFT; its checkpoint is not used for formal')))
    if template_path is not None:
        template = read(template_path)
        assert template == plan, 'Optional template must exactly match the derived minimal-change plan'
    # Validate all original/new inputs before the old owner is signaled. Only
    # the lifecycle is temporarily the old, currently held cycle for this check.
    module = F.install(plan)
    module.validate(plan)
    record(state / 'preflight-plan.json', plan)
    record(state / 'preflight-passed.json', dict(time=module.H.now(), old_owner_untouched=True,
        plan_sha256=sha(state / 'preflight-plan.json'), resume_receipt_sha256=sha(resume_path),
        completed_step=resumed['completed_step'], graphics_scope_reused=True))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready-manifest', type=Path, required=True)
    parser.add_argument('--resume-receipt', type=Path, required=True)
    parser.add_argument('--plan-template', type=Path)
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert Path(__file__).resolve() == (CODE / 'batch16_handoff.py').resolve()
    ready = ready_manifest(args.ready_manifest)
    old = read(OLD_OWNER / 'owner-plan.json')
    assert old['owner_dir'] == str(OLD_OWNER) and old['physical_gpus'] == [4, 5, 6, 7]
    assert sha(OLD_WRAPPER) == old['owner_script_sha256']
    old_wrapper = load('batch16_previous_formal_wrapper', OLD_WRAPPER)
    M = old_wrapper.install(old)
    M.validate(old, frozen=True)
    previous = read(OLD_OWNER / 'owner-identity.json')
    assert all(previous[k] == v for k, v in OLD_IDENTITY.items()) and M.H.same(previous)
    assert not (OLD_OWNER / 'final.json').exists() and not NEW_OWNER.exists()
    direct_sources = [Path(path) for path in old['source_sha256'] if Path(path).name == 'direct_continuation.py']
    assert len(direct_sources) == 1, 'Need the original pinned handoff helper source'
    direct_path = direct_sources[0]
    assert old['source_sha256'][str(direct_path)] == sha(direct_path)
    X = load('batch16_frozen_handoff_helpers', direct_path)
    X.OLD_OWNER, X.NEW_OWNER, X.OLD_IDENTITY = OLD_OWNER, NEW_OWNER, OLD_IDENTITY
    state = D / 'prepared'
    state.mkdir(mode=0o700, exist_ok=False)
    own_lock = (state / 'owner.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    me = M.H.proc(os.getpid())
    record(state / 'owner-identity.json', me)
    record(state / 'ready-used.json', dict(path=str(args.ready_manifest), sha256=sha(args.ready_manifest)))
    held, signaled, transferred, combined_module, combined_path = None, False, False, None, None
    failure, recovery, child = None, None, None
    try:
        plan = make_inputs(old, ready, args.resume_receipt, state, args.plan_template)
        prior_cycle = X.never_returned(old)
        support = OLD_WRAPPER.parent
        for name in ('rlt_returned_cycle.py', 'rlt_returned_multigpu_cycle.py'):
            assert old['source_sha256'][str(support / name)] == sha(support / name)
        R = load('batch16_adopt_child', support / 'rlt_returned_cycle.py')
        C = load('batch16_adopt_combined', support / 'rlt_returned_multigpu_cycle.py')
        base_567 = Path(read(Path(prior_cycle['children']['gpu567']['path']) / 'plan.json')['base_module'])
        assert sha(base_567) == R.BASE_SHA
        held = (Path(old['lifecycle_path']) / 'operation.lock').open('a')
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        X.never_returned(old)
        assert M.H.same(previous)
        intent = dict(kind='opendw-formal-adopt-held-rlt', parent_owner=str(OLD_OWNER), lifecycle_path=old['lifecycle_path'],
            owner_identity=previous, holder_identity=me, physical_gpus=[4, 5, 6, 7], time=M.H.now(),
            owner_plan_sha256=sha(OLD_OWNER / 'owner-plan.json'), combined_plan_sha256=sha(Path(old['lifecycle_path']) / 'plan.json'),
            child_plan_sha256={key: row['plan_sha256'] for key, row in prior_cycle['children'].items()},
            child_module_sha256={key: row['module_sha256'] for key, row in prior_cycle['children'].items()})
        intent_path = state / 'handoff-intent.json'
        record(intent_path, intent)
        # A checkpoint caller may have paused the coordinator. Its precise
        # driver is unpaused only so the old owner's SIGTERM cleanup can finish.
        driver = read(OLD_OWNER / 'formal/driver-identity.json')
        assert driver['namespace'] == next(row['namespace'] for row in old['trials'] if row['key'] == 'formal')
        assert M.H.same(driver)
        if M.H.proc(driver['pid'])['state'] in ('T', 't'):
            X.signal_exact(M, driver, signal.SIGCONT)
            record(state / 'driver-unpaused-for-cleanup.json', dict(time=M.H.now(), identity=driver))
        X.signal_exact(M, previous, signal.SIGTERM)
        signaled = True
        record(state / 'handoff-signal.json', dict(time=M.H.now(), identity=previous, signal=int(signal.SIGTERM)))
        deadline = time.monotonic() + 900
        while M.H.same(previous) or not (OLD_OWNER / 'final.json').is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError('Previous WM cleanup exceeded 900 seconds')
            M.atomic(state / 'status.json', dict(time=M.H.now(), phase='waiting_exact_old_cleanup'))
            time.sleep(2)
        X.certify_release(M, old, state)
        combined_path = state / 'adopted-cycle'
        scope = plan['graphics_scope']
        active = read(scope['activation_receipt'])
        children = {}
        for key in ('gpu4', 'gpu567'):
            prior = prior_cycle['children'][key]
            target = state / ('adopted-' + key)
            R.prepare_adopted(target, Path(prior['path']), Path(prior['module']), OLD_OWNER, base_567,
                scope_activation=scope['activation_receipt'], scope_manifest=active['manifest'],
                scope_id=active['scope_id'], handoff_intent=intent_path)
            children[key] = dict(path=str(target), module=str(target / 'rlt_returned_cycle.py'))
        C.prepare_adopted(combined_path, children, handoff_intent=intent_path)
        combined_module = load('batch16_adopted_combined_live', combined_path / 'rlt_returned_multigpu_cycle.py')
        combined_module.install_helper(combined_path)
        adoption = combined_path / 'adopted.json'
        combined_module.verify_adoption(combined_path, adoption)
        plan.update(lifecycle_path=str(combined_path), lifecycle_module=str(combined_path / 'rlt_returned_multigpu_cycle.py'),
            borrow_adoption=dict(receipt=str(adoption), sha256=sha(adoption)),
            handoff_evidence=dict(owner_final_path=str(OLD_OWNER / 'final.json'), owner_final_sha256=sha(OLD_OWNER / 'final.json')))
        plan['source_sha256'][plan['lifecycle_module']] = sha(plan['lifecycle_module'])
        for row in children.values():
            stage = Path(row['path'])
            plan['source_sha256'].update(read(stage / 'plan.json')['frozen_files'])
            plan['source_sha256'][row['module']] = sha(row['module'])
        plan_path = state / 'plan.json'
        record(plan_path, plan)
        F = load('batch16_formal_launch_preflight', CODE / 'batch16_formal_owner.py')
        new_module = F.install(plan)
        new_module.validate(plan)
        argv = [plan['python'], '-u', '-B', str(CODE / 'batch16_formal_owner.py'), '--plan', str(plan_path), 'owner']
        with (state / 'formal-owner.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=plan['repo'], env=read(plan['environment_file']),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = M.H.proc(child.pid)
        assert identity and identity['uid'] == 20001
        record(state / 'formal-owner-launched.json', dict(time=M.H.now(), identity=identity, argv=argv,
            plan=str(plan_path), plan_sha256=sha(plan_path), resumed_checkpoint=plan['resume_checkpoint']))
        deadline = time.monotonic() + 120
        while not (NEW_OWNER / 'owner-identity.json').is_file():
            assert child.poll() is None, 'New owner exited before accepting held-RLT responsibility'
            if time.monotonic() >= deadline:
                raise TimeoutError('Batch-16 owner acceptance deadline')
            time.sleep(1)
        accepted = read(NEW_OWNER / 'owner-identity.json')
        assert all(accepted[k] == identity[k] for k in ('pid', 'uid', 'start')) and M.H.same(accepted)
        assert read(NEW_OWNER / 'owner-plan.json')['borrow_adoption'] == plan['borrow_adoption']
        transferred = True
        record(state / 'handoff-completed.json', dict(time=M.H.now(), status='batch16_smoke_owner_running',
            identity=accepted, rlt_remained_stopped=True, formal_waits_for_startup_smoke=True, plan=str(plan_path)))
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        record(state / 'failure.json', failure)
    finally:
        if signaled and not transferred:
            try:
                if child is not None and child.poll() is None:
                    ident = M.H.proc(child.pid)
                    X.signal_exact(M, ident, signal.SIGTERM)
                    try:
                        child.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        X.signal_exact(M, ident, signal.SIGKILL)
                        child.wait(timeout=15)
                if (NEW_OWNER / 'final.json').is_file() and read(NEW_OWNER / 'final.json').get('rlt_return_dispatched'):
                    recovery = dict(status='new_owner_returned_rlt', receipt=str(NEW_OWNER / 'final.json'))
                else:
                    managed = X.emergency_cleanup(M, old, state, OLD_IDENTITY, 'old')
                    if (NEW_OWNER / 'owner-plan.json').exists():
                        new_plan = read(NEW_OWNER / 'owner-plan.json')
                        new_identity = read(NEW_OWNER / 'owner-identity.json') if (NEW_OWNER / 'owner-identity.json').exists() else None
                        managed += X.emergency_cleanup(M, new_plan, state, new_identity, 'batch16')
                    assert not M.H.gpu_processes([4, 5, 6, 7]), 'WM contexts remain; RLT return stays blocked'
                    if combined_module is None:
                        M.load_lifecycle(old)
                    target, helper = ((combined_path, combined_module.H) if combined_module else (Path(old['lifecycle_path']), M.C.H))
                    release = state / 'supervisor-wm-release.json'
                    record(release, dict(time=M.H.now(), cycle_id=target.name, gpus=[4, 5, 6, 7], all_workers_stopped=True,
                        managed_processes=managed, terminal_status='failed', reason='batch16_handoff_failed'))
                    if held is not None:
                        fcntl.flock(held, fcntl.LOCK_UN)
                        held.close()
                        held = None
                    recovery = dict(status='rlt_return_dispatched', result=helper.resume(target, release))
                record(state / 'recovery.json', recovery)
            except BaseException as exc:
                recovery = dict(status='recovery_failed_requires_attention', type=type(exc).__name__, error=str(exc),
                    traceback=traceback.format_exc())
                record(state / 'recovery-failed.json', recovery)
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
            held.close()
        record(state / 'final.json', dict(time=M.H.now(), transferred=transferred, signaled_old_owner=signaled,
            failure=failure, recovery=recovery, original_run_untouched=not signaled))
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
