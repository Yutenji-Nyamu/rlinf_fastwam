"""Compose the independently frozen GPU4 and GPU5-7 checkpoint cycles.

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
CHILD_STOP_MARKERS = {'gpu4': 'clean-old-stopped.json', 'gpu567': 'old-stopped.json'}
CHILD_ATTEMPT_MARKERS = {'gpu4': 'clean-old-stop-attempt.json', 'gpu567': 'old-stop-attempt.json'}


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


def prepare(stage, children):
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


def stop(stage):
    plan = load_plan(stage)
    assert not (stage/'clean-old-stop-attempt.json').exists(), 'Inspect previous partial stop before retry'
    H.save(stage/'clean-old-stop-attempt.json', dict(time=H.now(), children=plan['children']))
    # GPU4's legacy protected check requires GPU5-7 still alive at its exact stop.
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
