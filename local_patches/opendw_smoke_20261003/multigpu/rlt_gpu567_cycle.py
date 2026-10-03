"""Exact SZ3 GPU5/6/7 RLT checkpoint borrow/return component.

The outer four-GPU owner launches WM and supplies its release receipt. This
component never manages GPU4 or the shared Ray lifecycle. Run on SZ3 as UID20001.
Prepare is read-only toward old runs; stop first retires the known monitor-only
owner, then reuses the reviewed multi-run next_six stop implementation.
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
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/data/chenyiteng')
RECOVERY = ROOT / 'deployment-20261003/rlt-step-timeout-recovery-v1'
SOURCE_PLAN = RECOVERY / 'plan.json'
WATCH = ROOT / 'deployment-20260927/rlt-six-task-watch/plan.json'
EXPECTED_HEAD = 'd48e4a18f560e648635de9bb9a37dc8101ba1697'
# Reviewed recovery_owner.py launches once, then only polls its three children.
EXPECTED_OWNER_SOURCE_SHA = '9c6a97945c111c0b8c367378e872aa584ce6fc4914d6dbab0bebe1598d476a18'
EXPECTED_OWNER = {'pid': 3476669, 'uid': 20001, 'start': 698348482}
GPUS = [5, 6, 7]
KEY_GPUS = {'gpu5': [5], 'gpu6': [6], 'gpu7': [7]}
SCRIPT_NAME = 'rlt_gpu567_cycle.py'
H = None


def _pidfd_syscall(number, *arguments):
    """Reviewed Linux x86_64 LP64 fallback; never fall back to bare-PID kill."""
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
    """CPU-only kernel support check before prepare; signal zero changes no state."""
    fd = pidfd_open(os.getpid())
    try:
        pidfd_send(fd, 0)
    finally:
        os.close(fd)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert {k: r['gpus'] for k, r in plan['runs'].items()} == KEY_GPUS
    assert sha(__file__) == plan['script_sha256'], 'Cycle script changed'
    for path_key, hash_key in (
            ('helper_path', 'helper_sha256'), ('next_six_ops', 'next_six_ops_sha256'),
            ('source_plan', 'source_plan_sha256'), ('owner_source', 'owner_source_sha256')):
        assert sha(plan[path_key]) == plan[hash_key], path_key + ' changed'
    assert plan['owner_source_sha256'] == EXPECTED_OWNER_SOURCE_SHA
    assert sha(plan['owner_receipt_path']) == plan['owner_receipt_sha256'], 'Old owner receipt changed'
    for row in plan['runs'].values():
        rt = Path(row['original_run']) / 'runtime'
        for name, digest in row['original_runtime_sha256'].items():
            assert sha(rt / name) == digest, 'Old runtime input changed: ' + str(rt / name)
    source_check(plan)
    return plan


def check_invocation(identity, source, action, key=None):
    argv = [os.fsdecode(x) for x in (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0') if x]
    assert argv.count(str(source['ops'])) == 1 and argv.count('--stage') == 1
    assert Path(argv[argv.index('--stage') + 1]).resolve() == RECOVERY.resolve()
    assert argv[-1:] == [action] if key is None else argv[-2:] == [action, key]


def validate_owner_identity(receipt, current):
    """No PID-only fallback, including when /proc has reused the PID."""
    assert all(receipt.get(k) == v for k, v in EXPECTED_OWNER.items()), 'Unexpected monitor owner receipt'
    assert current and current.get('state') not in ('Z', 'X'), 'Monitor owner is not live'
    assert all(current.get(k) == v for k, v in EXPECTED_OWNER.items()), 'Monitor owner identity changed'
    if receipt.get('match_cmdline'):
        assert current.get('cmdline_sha256') == receipt['cmdline_sha256'], 'Monitor owner invocation changed'
    return current


def watch_targets(watch, rows, allow_returned=False):
    selected = {}
    for key, row in rows.items():
        matches = [(name, r) for name, r in watch['runs'].items() if r['gpus'] == row['gpus']]
        assert len(matches) == 1, 'Missing or duplicate watch GPU route: ' + key
        name, target = matches[0]
        allowed = {(row['original_run'], row['original_namespace'])}
        if allow_returned:
            allowed.add((row['new_run'], row['namespace']))
        assert (target['run'], target['namespace']) in allowed, 'Watch route changed: ' + key
        selected[key] = (name, target)
    return selected


def require_checkpoint(run, cfg, repo):
    recovery = H.select_recovery(run, cfg, repo)
    assert recovery['mode'] == 'resume_checkpoint' and recovery['checkpoint'] is not None, \
        'A complete RLT checkpoint is required; no fresh restart'
    return recovery


def prepare(stage, helper_path):
    from omegaconf import OmegaConf
    assert_stage(stage)
    pidfd_probe()
    assert not (stage / 'plan.json').exists(), 'Do not replay prepare'
    source = read(SOURCE_PLAN)
    assert source['uid'] == 20001 and source['head'] == EXPECTED_HEAD
    assert {k: r['gpus'] for k, r in source['runs'].items()} == KEY_GPUS
    assert sha(source['ops']) == EXPECTED_OWNER_SOURCE_SHA, 'Unreviewed recovery owner source'
    owner_receipt = RECOVERY / 'owner-identity.json'
    owner = read(owner_receipt)
    current = validate_owner_identity(owner, H.proc(owner['pid']))
    check_invocation(owner, source, 'owner')
    owner.update(cmdline_sha256=current['cmdline_sha256'], match_cmdline=True)
    assert H.same(owner)
    live = H.actors(source)
    rows = {}
    staged = {}
    for key, old in source['runs'].items():
        gpu = old['gpus'][0]
        assert old.get('repo', source['repo']) == source['repo']
        original = Path(old['run'])
        rt = original / 'runtime'
        identity = read(rt / 'driver-identity.json')
        assert identity['uid'] == 20001 and H.same(identity) and identity['namespace'] == old['namespace']
        check_invocation(identity, source, 'driver', key)
        identity.update(cmdline_sha256=H.proc(identity['pid'])['cmdline_sha256'], match_cmdline=True)
        cfg = H.config(rt / 'resolved.yaml')
        tb = original / 'tensorboard/config.yaml'
        if not tb.is_file():
            tb = original / original.name / 'tensorboard/config.yaml'
        assert cfg == H.config(tb), 'Actual/runtime config mismatch: ' + key
        assert cfg['cluster']['component_placement'] == {'actor,env,rollout': gpu}
        recovery = require_checkpoint(original, cfg, source['repo'])
        selected = H.active(live, old['namespace'])
        jobs = {r['job_id'] for r in selected}
        assert selected and len(jobs) == 1, 'Expected exactly one live old job: ' + key
        H.validate_actor_rows(selected, old['namespace'], jobs)
        tree = H.process_tree({identity['pid']} | {r['pid'] for r in selected if r.get('pid')})
        assert {r['pid'] for r in H.gpu_processes([gpu])} <= set(tree), 'Unrelated C/G context: ' + key
        new_run = original.with_name(original.name + '-after-' + stage.name)
        namespace = 'rlt-opendw-return-' + stage.name[-35:] + '-g' + str(gpu)
        assert not new_run.exists() and not H.active(live, namespace), 'Return target already exists'
        newcfg, changes = H.resumed_config(cfg, original, new_run, recovery['checkpoint']['path'])
        environment = read(rt / 'environment.json')
        assert not any(k in environment for k in H.MASKS), 'Discovery driver must not have a CUDA mask'
        environment = {k: v.replace(str(original), str(new_run)).replace(old['namespace'], namespace)
                       for k, v in environment.items()}
        rows[key] = dict(gpus=[gpu], original_run=str(original), original_namespace=old['namespace'],
            original_identity=identity, original_jobs=sorted(jobs),
            original_runtime_sha256={n: sha(rt / n) for n in ('resolved.yaml', 'environment.json', 'driver-identity.json')},
            new_run=str(new_run), namespace=namespace, entry=old['entry'], recovery=recovery,
            config_changes=changes, dependencies=H.dependency_snapshot(cfg),
            latest_metrics_before=H.latest_metrics(original))
        staged[key] = (cfg, newcfg, environment)
    watch_targets(read(WATCH), rows)
    plan = {k: source[k] for k in ('repo', 'head', 'uid', 'python', 'ray_address', 'ray_dashboard_url')}
    plan['source_sha256'] = {rel: sha(Path(source['repo']) / rel) for rel in H.CODE_FILES}
    source_check(plan)
    stage.mkdir(parents=True, mode=0o700, exist_ok=True)
    frozen = {}
    for name, src in (('rlt_checkpoint_lifecycle.py', helper_path),
                      ('next_six_ops.py', Path(source['repo']) / 'tools/next_six_20261002/ops.py'),
                      (SCRIPT_NAME, __file__)):
        dest = stage / name
        assert not dest.exists() and dest.resolve() != Path(src).resolve(), 'Freeze target exists'
        with dest.open('xb') as stream:
            stream.write(Path(src).read_bytes())
        dest.chmod(0o500)
        frozen[name] = dest
    for key, (cfg, newcfg, environment) in staged.items():
        pre = stage / 'prepared' / key
        pre.mkdir(parents=True, mode=0o700)
        OmegaConf.save(OmegaConf.create(cfg), pre / 'original.yaml', resolve=True)
        OmegaConf.save(OmegaConf.create(newcfg), pre / 'resolved.yaml', resolve=True)
        H.save(pre / 'environment.json', environment)
        rows[key]['prepared_sha256'] = {n: sha(pre / n) for n in ('original.yaml', 'resolved.yaml', 'environment.json')}
    plan.update(cycle_id=stage.name, time=H.now(), host=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        script_sha256=sha(__file__), helper_path=str(frozen['rlt_checkpoint_lifecycle.py']),
        helper_sha256=sha(frozen['rlt_checkpoint_lifecycle.py']), next_six_ops=str(frozen['next_six_ops.py']),
        next_six_ops_sha256=sha(frozen['next_six_ops.py']), source_plan=str(SOURCE_PLAN),
        source_plan_sha256=sha(SOURCE_PLAN), owner_source=source['ops'], owner_source_sha256=sha(source['ops']),
        owner_receipt_path=str(owner_receipt), owner_receipt_sha256=sha(owner_receipt), old_owner=owner,
        management_namespace='opendw-g567-rlt-ops-' + stage.name[-30:],
        gpu_processes_before=H.gpu_processes(list(range(8))), runs=rows)
    H.save(stage / 'plan.json', plan)
    H.save(stage / 'prepared.json', {'time': H.now(), 'gpus': GPUS, 'owner': owner,
        'runs': {k: {'new_run': r['new_run'], 'recovery': r['recovery'], 'config_changes': r['config_changes']}
                 for k, r in rows.items()}})
    return {'prepared': True, 'gpus': GPUS,
            'checkpoint_steps': {k: r['recovery']['checkpoint']['step'] for k, r in rows.items()}}


def retire_monitor(stage, plan):
    owner = plan['old_owner']
    done = stage / 'monitor-retired.json'
    if done.exists():
        assert read(done)['identity'] == owner and not H.same(owner), 'Retired monitor is live'
        return
    assert not (stage / 'monitor-retire-attempt.json').exists(), 'Ambiguous retirement; inspect exact identity'
    validate_owner_identity(owner, H.proc(owner['pid']))
    # pidfd prevents a recycled PID receiving a signal after this identity check.
    fd = pidfd_open(owner['pid'])
    try:
        validate_owner_identity(owner, H.proc(owner['pid']))
        H.save(stage / 'monitor-retire-attempt.json', {'time': H.now(), 'identity': owner, 'signal': 'SIGTERM'})
        pidfd_send(fd, signal.SIGTERM)
        for _ in range(100):
            if not H.same(owner):
                break
            time.sleep(0.1)
        assert not H.same(owner), 'Monitor did not retire; no child processes signaled'
    finally:
        os.close(fd)
    H.save(done, {'time': H.now(), 'identity': owner, 'monitor_only': True,
                  'driver_identities_unchanged': {k: H.same(r['original_identity']) for k, r in plan['runs'].items()}})


def stop(stage):
    plan = load_plan(stage)
    assert not (stage / 'rlt-stopped.json').exists() and not (stage / 'old-stop-attempt.json').exists(), \
        'Do not replay a stop attempt; use finalize-stopped after complete scoped stop only'
    watch_targets(read(WATCH), plan['runs'])
    live = H.actors(plan)
    bound = []
    for key, row in plan['runs'].items():
        assert H.same(row['original_identity']), 'Old driver identity changed: ' + key
        selected = H.active(live, row['original_namespace'])
        H.validate_actor_rows(selected, row['original_namespace'], set(row['original_jobs']))
        tree = H.process_tree({row['original_identity']['pid']} | {a['pid'] for a in selected if a.get('pid')})
        assert {r['pid'] for r in H.gpu_processes(row['gpus'])} <= set(tree), 'Foreign C/G context: ' + key
        pre = stage / 'prepared' / key
        assert sha(pre / 'original.yaml') == row['prepared_sha256']['original.yaml']
        require_checkpoint(row['original_run'], H.config(pre / 'original.yaml'), plan['repo'])
        bound.append(dict(run=row['original_run'], namespace=row['original_namespace'],
                          gpus=row['gpus'], identity=row['original_identity']))
    retire_monitor(stage, plan)
    ops = import_file('scoped_next_six_ops_gpu567', plan['next_six_ops'])
    ops.ST = stage

    def checked_stop():
        load_plan(stage)
        assert not H.same(plan['old_owner']), 'Old monitor is still live'
        return {**plan, 'old_runs': bound}

    def scoped_actors(_plan, namespace):
        matches = [r for r in plan['runs'].values() if r['original_namespace'] == namespace]
        assert len(matches) == 1, 'Unowned namespace'
        selected = H.active(H.actors(plan), namespace)
        H.validate_actor_rows(selected, namespace, set(matches[0]['original_jobs']))
        return selected

    ops.checked = checked_stop
    ops.active_actors = scoped_actors
    ops.gpu_pids = lambda gpus: sorted({r['pid'] for r in H.gpu_processes(gpus)})
    ops.stop_old(None)
    return finalize_stopped(stage)


def finalize_stopped(stage):
    """Freeze latest complete checkpoints after a completed, non-replayed stop."""
    from omegaconf import OmegaConf
    plan = load_plan(stage)
    if (stage / 'rlt-stopped.json').exists():
        return {'stopped': True, 'already_recorded': True}
    assert (stage / 'monitor-retired.json').is_file() and not H.same(plan['old_owner'])
    assert (stage / 'old-stop-attempt.json').is_file() and (stage / 'old-stopped.json').is_file()
    assert set(read(stage / 'old-stopped.json')['runs']) == {r['original_run'] for r in plan['runs'].values()}
    assert not H.gpu_processes(GPUS), 'Old C/G contexts remain on GPUs5/6/7'
    live = H.actors(plan)
    frozen = {}
    for key, row in plan['runs'].items():
        assert not H.same(row['original_identity']) and not H.active(live, row['original_namespace'])
        pre = stage / 'prepared' / key
        assert sha(pre / 'original.yaml') == row['prepared_sha256']['original.yaml']
        cfg = H.config(pre / 'original.yaml')
        recovery = require_checkpoint(row['original_run'], cfg, plan['repo'])
        newcfg, changes = H.resumed_config(cfg, row['original_run'], row['new_run'], recovery['checkpoint']['path'])
        OmegaConf.save(OmegaConf.create(newcfg), pre / 'resolved.yaml', resolve=True)
        metrics = H.latest_metrics(row['original_run'])
        rounds = metrics.get('env/success_once', {}).get('round')
        frozen[key] = dict(recovery=recovery, resolved_sha256=sha(pre / 'resolved.yaml'),
            config_changes=changes, last_metrics=metrics, collection_rounds_not_restored=None if rounds is None
            else max(0, rounds - recovery['checkpoint']['step']))
    before_pids = {r['pid'] for r in read(stage / 'old-stop-attempt.json')['processes']}
    gpu_after = H.gpu_processes(list(range(8)))
    old_remaining = [r for r in gpu_after if r['pid'] in before_pids]
    assert not old_remaining, 'Stopped old processes still have GPU contexts (including historical GPU0)'
    H.save(stage / 'rlt-stopped.json', dict(time=H.now(), cycle_id=stage.name, runs=frozen,
        all_original_drivers_stopped=True, all_original_namespaces_empty=True, gpus_released=GPUS,
        monitor_retired=True, old_owned_contexts_remaining=old_remaining, gpu_processes_after=gpu_after))
    return {'stopped': True, 'gpus': GPUS,
            'checkpoint_steps': {k: r['recovery']['checkpoint']['step'] for k, r in frozen.items()}}


def validate_release(stage, receipt_path):
    path = Path(receipt_path)
    assert path.resolve().is_relative_to(ROOT.resolve()) and path.stat().st_uid == 20001
    value = read(path)
    assert value['cycle_id'] == stage.name and value['gpus'] == GPUS
    assert value['terminal_status'] in ('completed', 'failed', 'timed_out', 'not_started')
    assert value['all_workers_stopped'] is True
    rows = value['managed_processes']
    assert isinstance(rows, list) and (rows or value['terminal_status'] == 'not_started')
    assert all(r['uid'] == 20001 and not H.same(r) for r in rows), 'Managed smoke process remains'
    # GPU4 may already be returned by the outer owner. It is outside this scope.
    assert not H.gpu_processes(GPUS), 'Smoke C/G context remains on GPUs5/6/7'
    return {'path': str(path), 'sha256': sha(path), 'terminal_status': value['terminal_status']}


def update_watch(plan, stage):
    assert not WATCH.is_symlink() and WATCH.stat().st_uid == 20001
    with (WATCH.parent / 'route.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = read(WATCH)
        after = copy.deepcopy(before)
        selected = watch_targets(after, plan['runs'], allow_returned=True)
        for key, (_, target) in selected.items():
            row = plan['runs'][key]
            target.update(run=row['new_run'], namespace=row['namespace'])
        if before == after:
            return
        if not (stage / 'watch-before.json').exists():
            H.save(stage / 'watch-before.json', before)
        H.atomic(WATCH, after)
        H.save(stage / 'watch-updated.json', {'time': H.now(), 'keys': sorted(selected),
            'targets': {k: t for k, (_, t) in selected.items()}})


def guard_names(plan, stage):
    path = ROOT / 'security/ray-guard/training_allowlist.json'
    assert not path.is_symlink() and path.stat().st_uid == 20001
    with (path.parent / 'recovery-allowlist.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = read(path)
        names = {r['namespace'] for r in plan['runs'].values()}
        if names <= set(before['namespaces']):
            return
        if not (stage / 'allowlist-before.json').exists():
            H.save(stage / 'allowlist-before.json', before)
        after = dict(before)
        after['namespaces'] = sorted(set(before['namespaces']) | names)
        H.atomic(path, after)
        H.save(stage / 'allowlist-added.json', {'time': H.now(), 'added': sorted(names - set(before['namespaces']))})


def install_helper(stage, helper_path=None):
    global H
    helper_path = helper_path or read(stage / 'plan.json')['helper_path']
    H = import_file('frozen_rlt_checkpoint_lifecycle_gpu567', helper_path)
    H.SCRIPT_NAME = SCRIPT_NAME
    H.load_plan, H.source_check = load_plan, source_check
    H.guard_names, H.update_watch, H.validate_release = guard_names, update_watch, validate_release


def status(stage):
    result = H.status(stage)
    plan = load_plan(stage)
    live = H.actors(plan)
    owned = set()
    for key, row in plan['runs'].items():
        selected = H.active(live, row['namespace'])
        jobs = {r['job_id'] for r in selected}
        assert len(jobs) <= 1, 'Multiple jobs in return namespace'
        H.validate_actor_rows(selected, row['namespace'], jobs)
        roots = {r['pid'] for r in selected if r.get('pid')}
        identity = result['runs'][key]['resume_identity']
        if identity and H.same(identity):
            roots.add(identity['pid'])
        owned.update(H.process_tree(roots))
    result['gpu_processes'] = H.gpu_processes(list(range(8)))
    result['secondary_contexts'] = [p for p in result['gpu_processes'] if p['pid'] in owned and p['gpu'] not in GPUS]
    result['old_monitor_alive'] = H.same(plan['old_owner'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle-dir', required=True)
    parser.add_argument('action', choices=('prepare', 'stop', 'finalize-stopped', 'resume', 'status', 'driver'))
    parser.add_argument('--helper', help='Existing audited rlt_cycle_sz3.py, prepare only')
    parser.add_argument('--key', choices=tuple(KEY_GPUS))
    parser.add_argument('--release-receipt')
    args = parser.parse_args()
    stage = Path(args.cycle_dir)
    assert_stage(stage)
    install_helper(stage, args.helper if args.action == 'prepare' else None)
    if args.action == 'driver':
        assert args.key
        H.driver(stage, args.key)
        return
    if args.action == 'status':
        result = status(stage)
        if result['all_first_rounds_verified'] and not (stage / 'first-round-verified.json').exists():
            H.save(stage / 'first-round-verified.json', result)
        print(json.dumps(result, ensure_ascii=False))
        return
    stage.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (stage / 'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            assert args.helper
            result = prepare(stage, args.helper)
        elif args.action == 'stop':
            result = stop(stage)
        elif args.action == 'finalize-stopped':
            result = finalize_stopped(stage)
        else:
            assert args.release_receipt
            result = H.resume(stage, args.release_receipt)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
