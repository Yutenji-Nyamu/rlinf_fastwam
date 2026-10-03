"""Compose returned-RLT cycles: both children reuse the GPU567 stop protocol.

The WM owner has one release receipt; each child retains its own repository,
checkpoint contract, namespaces and exact stop/return receipts. No shared Ray
restart, fresh RLT restart, or source-plan mutation is performed here.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import types

ROOT = Path('/data/chenyiteng')
H = None
CHILDREN = {}
CHILD_GPUS = {'gpu4': [4], 'gpu567': [5, 6, 7]}
CHILD_STOP_MARKERS = {'gpu4': 'old-stopped.json', 'gpu567': 'old-stopped.json'}
CHILD_ATTEMPT_MARKERS = {'gpu4': 'old-stop-attempt.json', 'gpu567': 'old-stop-attempt.json'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_path(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
    assert not path.is_symlink() and path.stat().st_uid == 20001
    return path


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_plan(stage):
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    stage = check_path(stage)
    plan = read(stage/'plan.json')
    assert plan['cycle_id'] == stage.name and plan['physical_gpus'] == [4, 5, 6, 7]
    assert plan['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert sha(__file__) == plan['script_sha256']
    assert set(plan['children']) == {'gpu4', 'gpu567'}
    merged = {}
    for key, frozen in plan['children'].items():
        child = check_path(frozen['path'])
        assert sha(child/'plan.json') == frozen['plan_sha256']
        assert sha(check_path(frozen['module'])) == frozen['module_sha256']
        subplan = CHILDREN[key].load_plan(child) if key in CHILDREN else read(child/'plan.json')
        assert {k: row['gpus'] for k, row in subplan['runs'].items()} == {
            'gpu'+str(g): [g] for g in CHILD_GPUS[key]}
        assert subplan['python'] == plan['python']
        assert subplan['ray_address'] == plan['ray_address']
        assert subplan['ray_dashboard_url'] == plan['ray_dashboard_url']
        assert not (merged.keys() & subplan['runs'].keys())
        merged.update(subplan['runs'])
    assert merged == plan['runs'] and set(merged) == {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
    return plan


def prepare(stage, children, _handoff_intent=None):
    """Both child prepares must already have validated live owners and full CPs."""
    stage = Path(stage)
    assert stage.is_absolute() and stage.resolve().is_relative_to(ROOT.resolve())
    assert not stage.exists() and not stage.is_symlink()
    check_path(stage.parent)
    assert set(children) == {'gpu4', 'gpu567'}
    frozen, plans = {}, {}
    for key, value in children.items():
        path = check_path(value['path'])
        module = check_path(value['module'])
        component = import_file('prepare_'+key, module)
        component.install_helper(path)
        plans[key] = component.load_plan(path)
        frozen[key] = dict(path=str(path), module=str(module),
                           module_sha256=sha(module), plan_sha256=sha(path/'plan.json'))
        if _handoff_intent:
            adoption = plans[key]['adopted_from']
            assert Path(adoption['handoff_intent']).resolve() == Path(_handoff_intent).resolve()
            assert adoption['handoff_intent_sha256'] == sha(_handoff_intent)
            assert (path/'rlt-stopped.json').is_file() and (path/'adopted.json').is_file()
            assert not (path/'resumed-dispatched.json').exists()
        else:
            assert not (path/CHILD_ATTEMPT_MARKERS[key]).exists()
            assert not (path/CHILD_STOP_MARKERS[key]).exists()
            assert not (path/'monitor-retire-attempt.json').exists()
            assert not (path/'rlt-stopped.json').exists()
    helper = component.H
    stage.mkdir(mode=0o700)
    dest = stage/Path(__file__).name
    dest.write_bytes(Path(__file__).read_bytes())
    dest.chmod(0o500)
    plan = {key: plans['gpu4'][key] for key in ('uid', 'python', 'ray_address', 'ray_dashboard_url')}
    plan.update(cycle_id=stage.name, physical_gpus=[4, 5, 6, 7],
                script_sha256=sha(dest), children=frozen,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                runs={**plans['gpu4']['runs'], **plans['gpu567']['runs']})
    helper.save(stage/'plan.json', plan)
    return plan


def prepare_adopted(stage, children, handoff_intent):
    """Combine the exact already-stopped children; no new stop or return is issued."""
    stage, handoff_intent = Path(stage), check_path(handoff_intent)
    intent = read(handoff_intent)
    assert intent['kind'] == 'opendw-formal-adopt-held-rlt'
    assert intent['physical_gpus'] == [4, 5, 6, 7]
    plan = prepare(stage, children, _handoff_intent=handoff_intent)
    install_helper(stage)
    finalize_stopped(stage)
    parent = check_path(intent['parent_owner'])
    release = parent / 'smoke-release.json'
    assert read(release)['all_workers_stopped'] is True
    receipts = {key: {'path': str(Path(value['path'])/'adopted.json'),
        'sha256': sha(Path(value['path'])/'adopted.json')} for key,value in plan['children'].items()}
    H.save(stage/'adopted.json', {'time': H.now(), 'kind': intent['kind'], 'cycle_id': stage.name,
        'physical_gpus': [4, 5, 6, 7], 'rlt_remained_stopped': True,
        'handoff_intent': str(handoff_intent), 'handoff_intent_sha256': sha(handoff_intent),
        'parent_owner': str(parent), 'parent_release': str(release), 'parent_release_sha256': sha(release),
        'child_receipts': receipts})
    return plan


def verify_adoption(stage, receipt_path=None):
    """Verify inherited ownership from any successor process, without requiring the old holder PID."""
    stage = Path(stage)
    plan = load_plan(stage)
    receipt_path = check_path(receipt_path or stage/'adopted.json')
    assert receipt_path.resolve() == (stage/'adopted.json').resolve()
    receipt = read(receipt_path)
    assert receipt['kind'] == 'opendw-formal-adopt-held-rlt'
    assert receipt['cycle_id'] == stage.name and receipt['physical_gpus'] == [4, 5, 6, 7]
    assert receipt['rlt_remained_stopped'] is True
    intent_path = check_path(receipt['handoff_intent'])
    assert sha(intent_path) == receipt['handoff_intent_sha256']
    intent = read(intent_path)
    assert intent['kind'] == receipt['kind'] and intent['physical_gpus'] == [4, 5, 6, 7]
    parent = check_path(receipt['parent_owner'])
    assert parent.resolve() == Path(intent['parent_owner']).resolve()
    assert not H.same(intent['owner_identity']), 'Previous owner is still live'
    assert not (parent/'rlt-return-dispatched.json').exists()
    release_path = check_path(receipt['parent_release'])
    assert release_path.resolve() == (parent/'smoke-release.json').resolve()
    assert sha(release_path) == receipt['parent_release_sha256']
    release = read(release_path)
    assert release['gpus'] == [4, 5, 6, 7] and release['all_workers_stopped'] is True
    stopped = read(stage/'rlt-stopped.json')
    assert stopped['cycle_id'] == stage.name and stopped['gpus_released'] == [4, 5, 6, 7]
    assert stopped['all_original_drivers_stopped'] is True and stopped['all_original_namespaces_empty'] is True
    assert set(stopped['runs']) == set(plan['runs']) == {'gpu4', 'gpu5', 'gpu6', 'gpu7'}
    assert not (stage/'resumed-dispatched.json').exists() and not (stage/'return-started.json').exists()
    assert set(receipt['child_receipts']) == set(plan['children']) == {'gpu4', 'gpu567'}
    for key, frozen in plan['children'].items():
        child = Path(frozen['path'])
        child_receipt = receipt['child_receipts'][key]
        assert Path(child_receipt['path']).resolve() == (child/'adopted.json').resolve()
        assert sha(child/'adopted.json') == child_receipt['sha256']
        adoption = read(child/'adopted.json')
        assert adoption['cycle_id'] == child.name and adoption['physical_gpus'] == CHILD_GPUS[key]
        assert adoption['rlt_remained_stopped'] is True
        child_stop = read(child/'rlt-stopped.json')
        assert child_stop['cycle_id'] == child.name and child_stop['gpus_released'] == CHILD_GPUS[key]
        assert child_stop['all_original_drivers_stopped'] is True and child_stop['all_original_namespaces_empty'] is True
        assert child_stop['runs'] == {k: stopped['runs'][k] for k in ('gpu'+str(g) for g in CHILD_GPUS[key])}
        source = adoption['adopted_from']
        assert Path(source['handoff_intent']).resolve() == intent_path.resolve()
        assert source['handoff_intent_sha256'] == receipt['handoff_intent_sha256']
        for cycle in (child, Path(source['cycle'])):
            assert not (cycle/'resumed-dispatched.json').exists()
            assert not list(cycle.glob('*-launch-attempt.json')) and not list(cycle.glob('*-launched.json'))
        assert all(not H.same(row['original_identity']) for row in
                   read(child/'plan.json')['runs'].values())
    return receipt


def stop(stage):
    plan = load_plan(stage)
    assert not (stage/'clean-old-stop-attempt.json').exists(), 'Inspect previous partial stop before retry'
    H.save(stage/'clean-old-stop-attempt.json', dict(time=H.now(), children=plan['children']))
    # Retain deterministic GPU4-first ordering; both children use returned-cycle guards.
    for key in ('gpu4', 'gpu567'):
        CHILDREN[key].stop(Path(plan['children'][key]['path']))
    H.save(stage/'clean-old-stopped.json', dict(time=H.now(), all_child_stops_completed=True))
    return finalize_stopped(stage)


def finalize_stopped(stage):
    plan = load_plan(stage)
    if (stage/'rlt-stopped.json').exists():
        return read(stage/'rlt-stopped.json')
    runs = {}
    for key, value in plan['children'].items():
        child = Path(value['path'])
        assert (child/'rlt-stopped.json').exists(), 'Child stop is incomplete: '+key
        receipt = read(child/'rlt-stopped.json')
        assert receipt['cycle_id'] == child.name and receipt['gpus_released'] == CHILD_GPUS[key]
        assert receipt['all_original_drivers_stopped'] is True and receipt['all_original_namespaces_empty'] is True
        expected = {'gpu'+str(g) for g in CHILD_GPUS[key]}
        assert set(receipt['runs']) == expected and not (runs.keys() & expected)
        runs.update(receipt['runs'])
    assert not H.gpu_processes([4, 5, 6, 7])
    receipt = dict(time=H.now(), cycle_id=stage.name, runs=runs,
                   all_original_drivers_stopped=True, all_original_namespaces_empty=True,
                   gpus_released=[4, 5, 6, 7])
    H.save(stage/'rlt-stopped.json', receipt)
    return receipt


def _release_for_child(stage, key, release):
    plan = load_plan(stage)
    value = plan['children'][key]
    child = Path(value['path'])
    path = stage/(key+'-wm-release.json')
    subset = copy.deepcopy(release)
    subset['cycle_id'] = child.name
    subset['gpus'] = CHILD_GPUS[key]
    if path.exists():
        assert read(path) == subset
    else:
        H.save(path, subset)
    return child, path


def resume(stage, release_receipt):
    plan = load_plan(stage)
    release = read(check_path(release_receipt))
    stopped = read(stage/'rlt-stopped.json')
    assert stopped['cycle_id'] == stage.name and set(stopped['runs']) == set(plan['runs'])
    assert stopped['gpus_released'] == [4, 5, 6, 7]
    assert release['cycle_id'] == stage.name and release['gpus'] == [4, 5, 6, 7]
    assert release['all_workers_stopped'] is True
    assert all(row['uid'] == 20001 and not H.same(row) for row in release['managed_processes'])
    if not (stage/'return-started.json').exists():
        assert not H.gpu_processes([4, 5, 6, 7])
        H.save(stage/'return-started.json', dict(time=H.now(), release_sha256=sha(release_receipt)))
    else:
        assert read(stage/'return-started.json')['release_sha256'] == sha(release_receipt)
    results = {}
    for key in ('gpu4', 'gpu567'):
        child, path = _release_for_child(stage, key, release)
        results[key] = CHILDREN[key].H.resume(child, path)
    if not (stage/'resumed-dispatched.json').exists():
        H.save(stage/'resumed-dispatched.json', dict(time=H.now(), children=results))
    return results


def status(stage):
    plan = load_plan(stage)
    result = dict(time=H.now(), cycle_id=stage.name, runs={}, children={})
    for key, value in plan['children'].items():
        component = CHILDREN[key]
        # GPU567 adds context/monitor observations at module level. GPU4's
        # helper already wraps status with its graphics-scope verification.
        inspect = getattr(component, 'status', component.H.status)
        sub = inspect(Path(value['path']))
        assert set(sub['runs']) == {'gpu'+str(g) for g in CHILD_GPUS[key]}
        assert not (result['runs'].keys() & sub['runs'].keys())
        result['children'][key] = sub
        result['runs'].update(sub['runs'])
    result['all_first_rounds_verified'] = all(
        row['first_round_verified'] for row in result['runs'].values())
    return result


def recover_partial(stage, release_receipt):
    """Return only conclusively stopped child cycles after WM cleanup.

    A sibling still running RLT is not a failed WM cleanup, and is never stopped
    merely to make the combined release predicate true.
    """
    plan = load_plan(stage)
    assert (stage/'clean-old-stop-attempt.json').is_file() and not (stage/'rlt-stopped.json').exists()
    release = read(check_path(release_receipt))
    assert release['cycle_id'] == stage.name and release['all_workers_stopped'] is True
    assert release['gpus'] == [4, 5, 6, 7]
    assert all(row['uid'] == 20001 and not H.same(row) for row in release['managed_processes'])
    prior_path = stage/'partial-recovery.json'
    if prior_path.exists():
        prior = read(prior_path)
        assert prior['release_sha256'] == sha(release_receipt), 'Partial recovery receipt changed'
        assert not prior['errors'], 'Inspect ambiguous previous partial recovery before retry'
        return prior['children']
    result = {}
    errors = {}
    for key, value in plan['children'].items():
        child = Path(value['path'])
        component = CHILDREN[key]
        try:
            if not (child/'rlt-stopped.json').exists() and (child/CHILD_STOP_MARKERS[key]).exists():
                component.finalize_stopped(child)
            if (child/'rlt-stopped.json').exists():
                child, receipt = _release_for_child(stage, key, release)
                result[key] = component.H.resume(child, receipt)
            else:
                sub = component.load_plan(child)
                assert all(component.H.same(row['original_identity']) for row in sub['runs'].values()), \
                    'Some original drivers stopped without a conclusive child receipt'
                result[key] = {'original_drivers_still_running': True, 'return_not_needed': True,
                               'monitor_retired': (child/'monitor-retired.json').exists()}
        except Exception as exc:
            errors[key] = {'type': type(exc).__name__, 'error': str(exc)}
    H.save(stage/'partial-recovery.json', dict(time=H.now(), children=result, errors=errors,
                                              release_sha256=sha(release_receipt)))
    if errors:
        raise RuntimeError('Partial recovery remains ambiguous: '+json.dumps(errors))
    return result


def install_helper(stage):
    global H, CHILDREN
    CHILDREN = {}
    plan = load_plan(stage)
    for key, frozen in plan['children'].items():
        module = import_file('frozen_multigpu_'+key, frozen['module'])
        module.install_helper(Path(frozen['path']))
        CHILDREN[key] = module
    # Copy the helper namespace, not the helper module itself: child resume/status
    # functions must retain their own globals and exact repository contracts.
    H = types.SimpleNamespace(**vars(CHILDREN['gpu4'].H))
    H.resume, H.status = resume, status
    load_plan(stage)
