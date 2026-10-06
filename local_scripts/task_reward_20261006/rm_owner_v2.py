"""One GPU4 RM command sequence; reuse exact RLT borrowing and Ray cleanup.

API (CPU-only preparation after the v1 RM owner has returned GPU4 RLT):
  rm_owner.py prepare --spec /.../task-reward-v2/prepared/spec.json
  rm_owner.py launch

Spec: commands[{key, kind: cpu|gpu|native, argv, cwd, timeout_seconds,
               result_file?, result_checks?, environment?,
               namespace?, config?, environment_fragment?}],
      environment{}, source_sha256{absolute_file: sha256}.
Native commands use run_native_eval.py's --receipt-dir/--namespace protocol.
Preparation/launch never edit the original WM/RLT plans or shared Ray runtime.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path('/data/chenyiteng')
S = ROOT / 'projects/opendw-robotwin-smoke-20261003'
D = S / 'task-reward-v2'
O = D / 'run'
PARENT = S / 'task-reward-v1/run'
PLAN = D / 'prepared/plan.json'
SCRIPT = D / 'code/rm_owner.py'
STAGE = D / 'prepared/rm1006-v2'
SCOPE = S / 'rynn-numeric-v1/rlt-return-repair-v2/scope'
MASKS = ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')
TOKEN, PHASE = 'OPENDW_SMOKE_OWNER_TOKEN', 'OPENDW_SMOKE_OWNER_PHASE'


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


def save(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def owned(path, exists=True):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
    assert not path.is_symlink()
    if exists:
        assert path.exists() and path.stat().st_uid == 20001
    return path


def check_host():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'


def arg(argv, flag):
    assert argv.count(flag) == 1, 'Expected one ' + flag
    return argv[argv.index(flag) + 1]


def check_environment(value):
    assert isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())
    assert not set(value).intersection((*MASKS, TOKEN, PHASE))


def validate_commands(commands, source, helper=None):
    assert isinstance(commands, list) and commands
    assert len({row['key'] for row in commands}) == len(commands)
    namespaces = []
    for row in commands:
        assert re.fullmatch(r'[a-z][a-z0-9_-]{0,39}', row['key'])
        assert row['kind'] in ('cpu', 'gpu', 'native')
        assert 0 < row['timeout_seconds'] <= 7 * 86400
        argv = row['argv']
        assert isinstance(argv, list) and argv and all(isinstance(a, str) and a for a in argv)
        interpreter = Path(argv[0])
        assert interpreter.is_absolute() and interpreter.is_file() and os.access(interpreter, os.X_OK)
        assert str(interpreter).startswith(('/data/chenyiteng/', '/home/chenyiteng/'))
        owned(row['cwd'])
        check_environment(row.get('environment', {}))
        for value in argv[1:]:
            if value.endswith('.py') and Path(value).is_absolute():
                assert source.get(value) == sha(owned(value)), 'Unfrozen command script: ' + value
        if row.get('result_file'):
            result = owned(row['result_file'], False)
            assert result.resolve().is_relative_to(D.resolve())
            assert not result.exists(), 'Result would overwrite an earlier attempt'
        assert isinstance(row.get('result_checks', {}), dict)
        assert not row.get('result_checks') or row.get('result_file')
        if row['kind'] != 'native':
            assert 'namespace' not in row
            continue
        namespace = row['namespace']
        assert re.fullmatch(r'opendw_rm_[a-zA-Z0-9_]+', namespace)
        namespaces.append(namespace)
        assert arg(argv, '--namespace') == namespace
        assert arg(argv, '--receipt-dir') == str(O / row['key'])
        assert arg(argv, '--config') == row['config']
        assert arg(argv, '--environment-fragment') == row['environment_fragment']
        for name in ('config', 'environment_fragment'):
            assert source.get(row[name]) == sha(owned(row[name])), 'Unfrozen native input'
        fragment = read(row['environment_fragment'])
        check_environment(fragment)
        assert fragment['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'] == str(SCOPE / 'scope.json')
        if helper:
            cfg = helper.config(row['config'])
            placement = cfg['cluster']['component_placement']
            assert placement and all(str(value) == '4' for value in placement.values()), 'Native workers must stay on GPU4'
            assert cfg['runner']['task_type'] == 'embodied_eval' and cfg['runner']['only_eval'] is True
            assert 'train' not in cfg['env']
    assert len(namespaces) == len(set(namespaces))


def short_prepare(module):
    source = inspect.getsource(module.prepare)
    before = "        new_run = original.with_name(original.name + '-after-' + stage.name)"
    after = ("        new_run = Path('/data/chenyiteng/results/rlinf-rlt') / "
             "('rlt-sz3-g' + str(gpu) + '-after-' + stage.name)\n"
             "        assert len(new_run.name.encode()) < 100")
    assert source.count(before) == 1
    exec(compile(source.replace(before, after, 1), str(__file__) + ':short_prepare', 'exec'), module.__dict__)


def single_parent_complete(owner, child_path, child_module, helper=None):
    """Validate v1's actual single-card receipts for prepare only.

    Evidence aliases match returned_cycle.prepare's frozen-file interface;
    there is no fabricated combined cycle. Stop/resume logic stays unchanged.
    """
    owner, child_path, child_module = owned(owner), owned(child_path), owned(child_module)
    assert owner == PARENT
    p, f = read(owner / 'owner-plan.json'), read(owner / 'final.json')
    assert p['owner_dir'] == str(owner) and p['mode'] == 'task-rm-single-gpu'
    assert p['physical_gpus'] == f['physical_gpus'] == [4]
    assert p['untouched_gpus'] == f['untouched_gpus'] == [5, 6, 7]
    assert f['terminal_status'] in ('completed', 'failed', 'timed_out')
    if f['terminal_status'] == 'completed':
        assert f['error'] is None
    else:
        assert isinstance(f['error'], dict) and f['error']['type'] and f['error']['error']
    assert f['recovery_error'] is None and f['rlt_borrowed'] is True and f['rlt_return_dispatched'] is True
    assert p['lifecycle_path'] == f['lifecycle_path'] == str(child_path)
    assert p['lifecycle_module'] == str(child_module)
    assert p['source_sha256'][str(child_path / 'plan.json')] == sha(child_path / 'plan.json')
    assert p['source_sha256'][str(child_module)] == sha(child_module)
    cp = read(child_path / 'plan.json')
    assert cp['group'] == 'gpu4' and set(cp['runs']) == {'gpu4'}
    assert cp['cycle_id'] == child_path.name and cp['runs']['gpu4']['gpus'] == [4]
    ident = read(owner / 'owner-identity.json')
    assert ident['uid'] == 20001 and ident['pid'] > 0 and ident['start'] > 0
    assert read(owner / 'cleanup.json')['all_stopped'] is True
    release_path = owner / 'rm-release.json'
    release = read(release_path)
    assert release['cycle_id'] == child_path.name and release['gpus'] == [4]
    assert release['terminal_status'] == f['terminal_status'] and release['all_workers_stopped'] is True
    assert release['managed_processes'] and all(row['uid'] == 20001 for row in release['managed_processes'])
    verified = read(child_path / 'dojo-release-verified.json')
    assert verified['path'] == str(release_path) and verified['sha256'] == sha(release_path)
    returned = read(child_path / 'resumed-dispatched.json')
    assert returned['cycle_id'] == child_path.name and set(returned['runs']) == {'gpu4'}
    assert returned['release'] == verified and returned['runs']['gpu4']['run'] == cp['runs']['gpu4']['new_run']
    owner_return = read(owner / 'rlt-return-dispatched.json')
    assert owner_return['lifecycle_path'] == str(child_path)
    assert owner_return['result']['resumed_dispatched'] is True
    assert owner_return['result']['cycle_id'] == child_path.name
    if helper is not None:
        assert not helper.same(ident), 'Previous single-card owner is still live'
        assert all(not helper.same(row) for row in release['managed_processes'])
    paths = dict(owner_plan=owner / 'owner-plan.json', final=owner / 'final.json',
        identity=owner / 'owner-identity.json', cleanup=owner / 'cleanup.json', release=release_path,
        owner_return=owner / 'rlt-return-dispatched.json', return_started=child_path / 'dojo-release-verified.json',
        combined_plan=child_path / 'plan.json', combined_return=child_path / 'resumed-dispatched.json',
        child_return=child_path / 'resumed-dispatched.json')
    evidence = {}
    for key, path in paths.items():
        evidence[key], evidence[key + '_sha256'] = str(path), sha(path)
    return 'gpu4', ident, evidence


def prepare_return_adapter(previous_module):
    """Freeze the single-card parent validator into a separate copied source."""
    source = previous_module.read_text()
    tree = ast.parse(source, filename=str(previous_module))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parent_complete']
    assert len(functions) == 1
    original = functions[0]
    replacement = inspect.getsource(single_parent_complete)
    replacement = replacement.replace('def single_parent_complete(', 'def parent_complete(', 1)
    assert replacement.count('assert owner == PARENT') == 1
    replacement = replacement.replace('assert owner == PARENT', "assert owner == Path(" + repr(str(PARENT)) + ")", 1)
    lines = source.splitlines(keepends=True)
    revised = ''.join(lines[:original.lineno - 1]) + replacement + '\n' + ''.join(lines[original.end_lineno:])
    parsed = ast.parse(revised, filename='rlt_returned_cycle.py')
    # All other top-level code, including stop/resume and load_plan, is exact.
    old_other = [ast.dump(node, include_attributes=False) for node in tree.body if node is not original]
    new_other = [ast.dump(node, include_attributes=False) for node in parsed.body
                 if not (isinstance(node, ast.FunctionDef) and node.name == 'parent_complete')]
    assert old_other == new_other
    folder = D / 'prepared/returned-cycle-adapter'
    assert not folder.exists()
    folder.mkdir(mode=0o700)
    target = folder / 'rlt_returned_cycle.py'
    with target.open('x') as stream:
        stream.write(revised)
    target.chmod(0o500)
    receipt = folder / 'source-delta.json'
    save(receipt, dict(original=str(previous_module), original_sha256=sha(previous_module),
        adapted=str(target), adapted_sha256=sha(target), parent_owner=str(PARENT),
        change='Replace only parent_complete with exact GPU4 RM release validation',
        other_top_level_code_unchanged=True, scope='prepare provenance; original stop/resume/load_plan unchanged'))
    return target, receipt


def prepare(spec_path):
    check_host()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Prepare with CUDA hidden'
    assert not PLAN.exists() and not O.exists() and not STAGE.exists()
    assert Path(__file__).resolve() == SCRIPT.resolve()
    spec_path = owned(spec_path)
    spec = read(spec_path)
    source = dict(spec['source_sha256'])
    assert source and all(sha(owned(path)) == digest for path, digest in source.items())
    check_environment(spec['environment'])
    validate_commands(spec['commands'], source)
    old = read(PARENT / 'owner-plan.json')
    previous, module_path = owned(old['lifecycle_path']), owned(old['lifecycle_module'])
    assert sha(previous / 'plan.json') == old['source_sha256'][str(previous / 'plan.json')]
    assert sha(module_path) == old['source_sha256'][str(module_path)]
    single_parent_complete(PARENT, previous, module_path)
    adapter_path, adapter_receipt = prepare_return_adapter(module_path)
    R = load('rm_prepare_returned_cycle', adapter_path)
    R.parent_complete(PARENT, previous, module_path)
    prior_plan = read(previous / 'plan.json')
    active = read(SCOPE / 'activation.json')
    assert active['status'] == 'active' and active['manifest'] == str(SCOPE / 'scope.json')
    assert sha(active['manifest']) == active['manifest_sha256']
    assert prior_plan['scope_manifest'] == active['manifest']
    short_prepare(R)
    R.prepare(STAGE, previous, module_path, PARENT, owned(prior_plan['base_module']),
        scope_activation=str(SCOPE / 'activation.json'), scope_manifest=active['manifest'], scope_id=active['scope_id'])
    cp = R.load_plan(STAGE)
    validate_commands(spec['commands'], source, R.H)
    catalog_path = owned(old['catalog_module'])
    assert old['source_sha256'][str(catalog_path)] == sha(catalog_path)
    M = load('rm_prepare_process_catalog', catalog_path)
    M.H = R.H
    M.pidfd_probe()
    untouched = copy.deepcopy(old['untouched_drivers'])
    for key, row in untouched.items():
        assert R.H.same(row['identity'])
        identity = read(Path(row['run']) / 'runtime/driver-identity.json')
        assert all(identity[field] == row['identity'][field] for field in ('pid', 'uid', 'start'))
        assert identity['namespace'] == row['namespace']
    assert set(untouched) == {'gpu5', 'gpu6', 'gpu7'}
    for row in spec['commands']:
        if row['kind'] == 'native':
            assert not R.H.active(R.H.actors(cp), row['namespace'])
    files = [SCRIPT, spec_path, STAGE / 'plan.json', catalog_path, adapter_path, adapter_receipt, SCOPE / 'activation.json',
             Path(active['manifest']), Path(active['runtime_path']), Path(active['environment_fragment_file'])]
    source.update(cp['frozen_files'])
    source.update({str(path): sha(path) for path in files})
    plan = dict(mode='task-rm-single-gpu', owner_dir=str(O), parent_owner=str(PARENT),
        lifecycle_path=str(STAGE), lifecycle_module=str(STAGE / 'rlt_returned_cycle.py'),
        catalog_module=str(catalog_path), source_sha256=source, python=cp['python'],
        physical_gpus=[4], untouched_gpus=[5, 6, 7], untouched_drivers=untouched,
        commands=copy.deepcopy(spec['commands']), environment=copy.deepcopy(spec['environment']),
        ray_address=cp['ray_address'], ray_dashboard_url=cp['ray_dashboard_url'],
        management_namespace='opendw_rm_ops_1006_v2',
        trials=[dict(key=row['key'], namespace=row['namespace']) for row in spec['commands'] if row['kind'] == 'native'])
    save(PLAN, plan)
    save(D / 'prepared.json', dict(plan=str(PLAN), plan_sha256=sha(PLAN), preparation_only=True,
        gpu4_checkpoint=cp['runs']['gpu4']['recovery']['checkpoint']['step'], untouched_gpus=[5, 6, 7]))
    print(json.dumps(read(D / 'prepared.json')), flush=True)


def modules():
    check_host()
    plan = read(PLAN)
    assert read(D / 'prepared.json')['plan_sha256'] == sha(PLAN)
    assert plan['mode'] == 'task-rm-single-gpu' and plan['owner_dir'] == str(O)
    assert plan['parent_owner'] == str(PARENT) and plan['physical_gpus'] == [4]
    assert plan['untouched_gpus'] == [5, 6, 7] and set(plan['untouched_drivers']) == {'gpu5', 'gpu6', 'gpu7'}
    assert plan['lifecycle_path'] == str(STAGE)
    assert plan['lifecycle_module'] == str(STAGE / 'rlt_returned_cycle.py')
    assert plan['source_sha256'][str(SCRIPT)] == sha(__file__)
    for path, digest in plan['source_sha256'].items():
        assert sha(owned(path)) == digest, 'Frozen input changed: ' + path
    R = load('rm_runtime_returned_cycle', plan['lifecycle_module'])
    R.install_helper(STAGE)
    cp = R.load_plan(STAGE)
    assert cp['group'] == 'gpu4' and set(cp['runs']) == {'gpu4'}
    assert cp['completed_owner'] == str(PARENT) and cp['python'] == plan['python']
    assert cp['ray_address'] == plan['ray_address'] and cp['ray_dashboard_url'] == plan['ray_dashboard_url']
    M = load('rm_runtime_process_catalog', plan['catalog_module'])
    M.H, M.GPUS = R.H, [4]
    M.pidfd_probe()
    check_environment(plan['environment'])
    validate_commands(plan['commands'], plan['source_sha256'], R.H)
    return plan, R, M


def launch():
    plan, R, _ = modules()
    assert not O.exists() and not (D / 'launch-attempt.json').exists()
    cp = R.load_plan(STAGE)
    assert R.H.same(cp['runs']['gpu4']['original_identity'])
    assert not (STAGE / 'old-stop-attempt.json').exists()
    assert all(R.H.same(row['identity']) for row in plan['untouched_drivers'].values())
    save(D / 'launch-attempt.json', dict(time=R.H.now(), plan_sha256=sha(PLAN)))
    argv = [plan['python'], '-u', '-B', str(SCRIPT), 'owner']
    environment = dict(plan['environment'], CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    with (D / 'owner-launch.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=str(D / 'code'), env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = R.H.proc(child.pid)
    assert identity and identity['uid'] == 20001
    save(D / 'launch-receipt.json', dict(time=R.H.now(), identity=identity, argv=argv, plan_sha256=sha(PLAN)))
    print(json.dumps(read(D / 'launch-receipt.json')), flush=True)


def environment_for(plan, row, token):
    environment = dict(plan['environment'], **row.get('environment', {}))
    if row['kind'] == 'native':
        environment.update(read(row['environment_fragment']))
        environment.update(RAY_ADDRESS=plan['ray_address'], CLUSTER_NAMESPACE=row['namespace'])
        for key in MASKS:
            environment.pop(key, None)
    else:
        environment['CUDA_VISIBLE_DEVICES'] = '4' if row['kind'] == 'gpu' else ''
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', **{TOKEN: token, PHASE: row['key']})
    return environment


def snapshot(H, catalog, row):
    catalog.scan()
    live = catalog.live()
    pids = {identity['pid'] for identity in live}
    contexts = H.gpu_processes(list(range(8)))
    ours = [item for item in contexts if item['pid'] in pids]
    assert not [item for item in ours if item['gpu'] != 4], 'RM child escaped GPU4'
    assert not [item for item in contexts if item['gpu'] == 4 and item['pid'] not in pids], 'Foreign GPU4 context; no cleanup authorized'
    if row['kind'] == 'cpu':
        assert not ours, 'CPU command initialized a GPU context'
    memory = subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20).strip()
    value = dict(time=H.now(), phase=row['key'], owned_gpu_contexts=ours,
        gpu4_memory_used_total_mib_utilization_pct=memory, live_processes=live)
    H.atomic(O / 'heartbeat.json', value)
    with (O / 'resources.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def command_result(row):
    if not row.get('result_file'):
        return None
    path = owned(row['result_file'])
    value = read(path)
    for key, expected in row.get('result_checks', {}).items():
        actual = value
        for part in key.split('.'):
            actual = actual[part]
        assert actual == expected, 'Command result failed: ' + key
    return dict(path=str(path), sha256=sha(path))


def owner():
    import fcntl
    plan, R, M = modules()
    H = R.H
    O.mkdir(mode=0o700)
    H.save(O / 'owner-plan.json', plan)
    H.save(O / 'owner-identity.json', H.proc(os.getpid()))
    catalog = M.Catalog(O / 'processes.json', uuid.uuid4().hex)
    closing = dict(value=False, critical=False, signal=None)
    children, results = [], []
    active_command, active_child, started = None, None, None
    borrowed = returned = False
    status, error, recovery_error = 'not_started', None, None
    lock = (STAGE / 'operation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def terminate(signum, _frame):
        closing['signal'] = signum
        if not closing['value'] and not closing['critical']:
            raise RuntimeError('RM owner received signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        # The small CPU commands also run after borrowing, keeping one simple
        # lifecycle and permitting exact GPU4 availability checks at every step.
        closing['critical'] = True
        try:
            R.stop(STAGE)
            assert (STAGE / 'rlt-stopped.json').is_file()
            borrowed = True
        finally:
            closing['critical'] = False
        H.save(O / 'borrowed.json', dict(time=H.now(), physical_gpus=[4], cycle_id=STAGE.name))
        if closing['signal']:
            raise RuntimeError('Termination requested during borrowing')
        assert not H.gpu_processes([4])
        if plan['trials']:
            M.add_allowlist(plan)
        for row in plan['commands']:
            target = O / row['key']
            target.mkdir(mode=0o700)
            active_command, active_child, started = row, None, time.monotonic()
            if row['kind'] == 'native':
                assert not H.active(H.actors(plan), row['namespace'])
            H.save(target / 'launch-attempt.json', dict(time=H.now(), argv=row['argv'], timeout_seconds=row['timeout_seconds']))
            closing['critical'] = True
            try:
                with (target / 'command.log').open('x') as log:
                    child = subprocess.Popen(row['argv'], cwd=row['cwd'],
                        env=environment_for(plan, row, catalog.token), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active_child = child
                children.append(child)
                identity = H.proc(child.pid)
                assert identity and identity['uid'] == 20001
                catalog.rows[(identity['pid'], identity['start'])] = dict(identity, phase=row['key'], proof='exact RM owner Popen child')
                catalog.write()
                H.save(target / 'child-identity.json', identity)
            finally:
                closing['critical'] = False
            if closing['signal']:
                raise RuntimeError('Termination requested during command launch')
            status = 'failed'
            while child.poll() is None:
                if row['kind'] == 'native':
                    M.register_actors(plan, row, catalog)
                snapshot(H, catalog, row)
                H.atomic(O / 'state.json', dict(time=H.now(), phase=row['key'],
                    elapsed_seconds=time.monotonic() - started))
                if time.monotonic() - started >= row['timeout_seconds']:
                    status = 'timed_out'
                    raise TimeoutError('Command timed out: ' + row['key'])
                time.sleep(10)
            cleaned = M.cleanup(plan, catalog, row['key'])
            H.save(target / 'cleanup.json', cleaned)
            result = dict(key=row['key'], exit_code=child.returncode, seconds=time.monotonic() - started)
            if child.returncode == 0:
                result['output'] = command_result(row)
            H.save(target / 'command-result.json', result)
            results.append(result)
            assert child.returncode == 0, 'RM command failed: ' + row['key']
            if row['kind'] == 'native':
                receipt = read(target / 'ray-job.json')
                assert receipt['namespace'] == row['namespace'] and receipt['job_id']
            assert not H.gpu_processes([4]), 'GPU4 context remains after command cleanup'
            active_command = None
        status = 'completed'
    except BaseException as exc:
        error = dict(type=type(exc).__name__, error=str(exc))
        if active_command is not None:
            path = O / active_command['key'] / 'command-result.json'
            if not path.exists():
                result = dict(key=active_command['key'], terminal_status=status, error=error,
                    exit_code=active_child.poll() if active_child is not None else None,
                    seconds=time.monotonic() - started)
                H.save(path, result)
                results.append(result)
    finally:
        closing['value'] = True
        try:
            cleared = M.cleanup(plan, catalog)
            H.save(O / 'cleanup.json', cleared)
            for child in children:
                child.wait(timeout=5)
            if not (STAGE / 'rlt-stopped.json').exists() and (STAGE / 'old-stopped.json').exists():
                R.finalize_stopped(STAGE)
            borrowed = (STAGE / 'rlt-stopped.json').exists()
            if borrowed:
                assert not H.gpu_processes([4]), 'GPU4 compute/graphics context remains'
                known = {item['pid'] for item in catalog.rows.values()}
                assert not [item for item in H.gpu_processes(list(range(8))) if item['pid'] in known]
                release = O / 'rm-release.json'
                H.save(release, dict(time=H.now(), cycle_id=STAGE.name, gpus=[4], terminal_status=status,
                    all_workers_stopped=True, managed_processes=list(catalog.rows.values())))
                result = H.resume(STAGE, release)
                returned = True
                H.save(O / 'rlt-return-dispatched.json', dict(time=H.now(), result=result, lifecycle_path=str(STAGE)))
            elif (STAGE / 'old-stop-attempt.json').exists():
                raise RuntimeError('GPU4 borrow incomplete; inspect the exact existing lifecycle')
        except BaseException as exc:
            recovery_error = dict(type=type(exc).__name__, error=str(exc))
        H.save(O / 'final.json', dict(time=H.now(), terminal_status=status, commands=results, error=error,
            recovery_error=recovery_error, physical_gpus=[4], untouched_gpus=[5, 6, 7],
            rlt_borrowed=borrowed, rlt_return_dispatched=returned, lifecycle_path=str(STAGE),
            first_round_validation_pending=returned,
            untouched_drivers_live={key: H.same(row['identity']) for key, row in plan['untouched_drivers'].items()}))
        lock.close()
    return 0 if status == 'completed' and recovery_error is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'owner'))
    parser.add_argument('--spec', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        assert args.spec is not None
        prepare(args.spec)
    else:
        assert args.spec is None
        return globals()[args.action]()


if __name__ == '__main__':
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
