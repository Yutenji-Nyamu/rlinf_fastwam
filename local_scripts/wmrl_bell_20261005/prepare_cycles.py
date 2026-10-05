"""Prepare fresh SZ3 RLT checkpoint cycles for WM priority; never stop/start jobs.

GPU4 comes from the completed numeric return repair, GPU567 from Rynn v2's
normal four-card return. The original stop/resume modules are copied unchanged.
Only prepare's run-name expression is changed in memory to avoid ENAMETOOLONG.
Run with the existing RLT Python, CUDA_VISIBLE_DEVICES='', after root review.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import socket
import sys

ROOT = Path('/data/chenyiteng')
S = ROOT / 'projects/opendw-robotwin-smoke-20261003'
NUMERIC_OWNER = S / 'rynn-numeric-v1/run'
REPAIR = S / 'rynn-numeric-v1/rlt-return-repair-v2'
GPU4_STAGE = REPAIR / 'rynn-numeric-v1-gpu4'
FOUR_OWNER = S / 'runs/rynn-success-v2'
CHILD_SHA = '2667484807da8065ce523ea398b5502b384e39898de58b98d50aef72a71ab7b3'
COMBINED_SHA = '6d92df58de2f0d905273eedc879cc3cdc3d1523d58d5db57f4403e748fb961a1'
NAMED_SCOPE_SHA = '96d183e00506cd44345cfadcd61997f10ef8b4d810b9a2aee518ef2930b70114'


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
        assert path.stat().st_uid == 20001
    return path


def save_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def repaired_parent_complete(owner, child_path, child_module, helper=None):
    """Accept only the exact successful repair; preserve the failed original final."""
    owner, child_path, child_module = map(owned, (owner, child_path, child_module))
    assert owner == NUMERIC_OWNER and child_path == GPU4_STAGE
    assert child_module == child_path / 'rlt_returned_cycle.py'
    assert sha(child_module) == CHILD_SHA
    receipt_path = owned(REPAIR / 'repaired.json')
    receipt = read(receipt_path)
    assert receipt['lifecycle_path'] == str(child_path)
    assert receipt['result']['resumed_dispatched'] is True
    assert receipt['result']['cycle_id'] == child_path.name
    assert receipt['plan_sha256'] == sha(child_path / 'plan.json')
    assert receipt['return_sha256'] == sha(child_path / 'resumed-dispatched.json')
    assert receipt['original_final_preserved_sha256'] == sha(owner / 'final.json')
    p, f = read(owner / 'owner-plan.json'), read(owner / 'final.json')
    assert p['owner_dir'] == str(owner) and p['mode'] == 'rynn-single-gpu-diagnostic'
    assert p['physical_gpus'] == f['physical_gpus'] == [4]
    assert p['untouched_gpus'] == f['untouched_gpus'] == [5, 6, 7]
    assert f['terminal_status'] == 'completed' and f['child_exit_code'] == 0
    assert f['error'] is None and f['rlt_borrowed'] is True
    assert f['rlt_return_dispatched'] is False
    assert f['recovery_error']['type'] == 'OSError'
    assert '[Errno 36]' in f['recovery_error']['error']
    original_stage = S / 'rynn-numeric-v1/prepared/rynn-numeric-v1-gpu4'
    assert p['lifecycle_path'] == str(original_stage)
    assert p['source_sha256'][str(original_stage / 'plan.json')] == sha(original_stage / 'plan.json')
    assert p['source_sha256'][p['lifecycle_module']] == sha(p['lifecycle_module']) == CHILD_SHA
    cp = read(child_path / 'plan.json')
    assert cp['group'] == 'gpu4' and set(cp['runs']) == {'gpu4'}
    assert cp['cycle_id'] == child_path.name and cp['runs']['gpu4']['gpus'] == [4]
    assert cp['repair']['original_stage'] == str(original_stage)
    assert cp['repair']['no_new_stop'] is True
    assert cp['runs']['gpu4']['new_run'] == receipt['new_run']
    for path, digest in cp['repair']['evidence'].items():
        assert sha(owned(path)) == digest, 'Repair evidence changed: ' + path
    ident = read(owner / 'owner-identity.json')
    assert ident['uid'] == 20001 and ident['pid'] > 0 and ident['start'] > 0
    assert read(owner / 'cleanup.json')['all_stopped'] is True
    release_path = owner / 'diagnostic-release.json'
    release = read(release_path)
    assert release['cycle_id'] == child_path.name and release['gpus'] == [4]
    assert release['terminal_status'] == f['terminal_status']
    assert release['all_workers_stopped'] is True
    assert release['managed_processes'] and all(row['uid'] == 20001 for row in release['managed_processes'])
    verified = read(child_path / 'dojo-release-verified.json')
    assert verified['path'] == str(release_path) and verified['sha256'] == sha(release_path)
    returned = read(child_path / 'resumed-dispatched.json')
    assert returned['cycle_id'] == child_path.name and set(returned['runs']) == {'gpu4'}
    assert returned['release'] == verified
    assert returned['runs']['gpu4']['run'] == cp['runs']['gpu4']['new_run']
    launched = read(child_path / 'gpu4-launched.json')
    assert launched == receipt['gpu4_launch']
    assert launched['run'] == cp['runs']['gpu4']['new_run']
    assert launched['namespace'] == cp['runs']['gpu4']['namespace']
    assert launched['recovery_mode'] == 'resume_checkpoint'
    if helper is not None:
        assert not helper.same(ident), 'Original numeric owner is still live'
        assert all(not helper.same(row) for row in release['managed_processes'])
        assert helper.same(launched['identity']), 'Repaired return driver is not live'
    # These aliases satisfy the original preparer's frozen-evidence schema.
    # No combined cycle or successful original owner final is fabricated.
    paths = dict(owner_plan=owner / 'owner-plan.json', final=owner / 'final.json',
        identity=owner / 'owner-identity.json', cleanup=owner / 'cleanup.json', release=release_path,
        owner_return=receipt_path, return_started=child_path / 'dojo-release-verified.json',
        combined_plan=child_path / 'plan.json', combined_return=child_path / 'resumed-dispatched.json',
        child_return=child_path / 'gpu4-launched.json')
    evidence = {}
    for key, path in paths.items():
        evidence[key], evidence[key + '_sha256'] = str(path), sha(path)
    return 'gpu4', ident, evidence


def short_prepare(module):
    """Patch one preparation expression; copied runtime module remains byte-identical."""
    source = inspect.getsource(module.prepare)
    old = "        new_run = original.with_name(original.name + '-after-' + stage.name)"
    new = ("        new_run = Path('/data/chenyiteng/results/rlinf-rlt') / "
           "('rlt-sz3-g' + str(gpu) + '-after-' + stage.name)\n"
           "        assert len(new_run.name.encode()) < 100")
    assert source.count(old) == 1
    exec(compile(source.replace(old, new), str(Path(__file__)) + ':short_prepare', 'exec'), module.__dict__)


def named_scope():
    p = read(GPU4_STAGE / 'plan.json')
    manifest_path, active_path = owned(p['scope_manifest']), owned(p['scope_activation'])
    assert manifest_path == REPAIR / 'scope/scope.json'
    assert active_path == REPAIR / 'scope/activation.json'
    m, active = read(manifest_path), read(active_path)
    assert active['status'] == 'active' and active['uid'] == 20001
    assert active['manifest'] == str(manifest_path) and active['manifest_sha256'] == sha(manifest_path)
    assert p['scope_id'] == active['scope_id'] == m['token']
    assert m['physical_gpus'] == [4, 5, 6, 7] and m['cpu_full_mask_target'] == 4
    assert sha(owned(active['runtime_path'])) == active['runtime_sha256'] == NAMED_SCOPE_SHA
    runtime = load('bell_named_scope_readonly', active['runtime_path'])
    runtime.read_manifest(manifest_path)  # Rechecks live profile bytes and named settings.
    fragment = owned(active['environment_fragment_file'])
    assert sha(fragment) == active['environment_fragment_sha256']
    assert read(fragment) == active['environment_fragment']
    assert read(fragment)['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'] == str(manifest_path)
    return dict(scope_activation=str(active_path), scope_manifest=str(manifest_path), scope_id=active['scope_id'])


def pin_prepared(stage, module, scope):
    """Freeze this preparer and accepted scope in newly created files only."""
    path = stage / 'plan.json'
    p = read(path)
    assert not any((stage / name).exists() for name in ('rlt-stopped.json', 'old-stop-attempt.json', 'resumed-dispatched.json'))
    extras = [owned(__file__), Path(scope['scope_activation']), Path(scope['scope_manifest'])]
    active = read(scope['scope_activation'])
    extras += [owned(active['runtime_path']), owned(active['environment_fragment_file'])]
    p['frozen_files'].update({str(item): sha(item) for item in extras})
    p['prepare_only_changes'] = dict(short_run_names=True, original_runtime_module_sha256=CHILD_SHA,
        named_graphics_scope=str(scope['scope_manifest']), stop_or_start_performed=False)
    temporary = stage / 'plan.prepare-final.tmp'
    save_new(temporary, p)
    os.replace(temporary, path)
    module.load_plan(stage)
    assert sha(stage / 'rlt_returned_cycle.py') == CHILD_SHA
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--cycle-prefix', default='bell-v1')
    parser.add_argument('--groups', choices=('gpu4', 'all'), default='all')
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run preparation with empty CUDA visibility'
    assert re.fullmatch(r'[a-z][a-z0-9-]{0,19}', args.cycle_prefix)
    output = owned(args.prepared_dir, exists=False)
    assert output.resolve().is_relative_to(S.resolve()) and not output.exists(), 'Fresh preparation directory required'
    owned(output.parent)
    repaired_parent_complete(NUMERIC_OWNER, GPU4_STAGE, GPU4_STAGE / 'rlt_returned_cycle.py')
    scope = named_scope()
    owner_plan = read(FOUR_OWNER / 'owner-plan.json')
    combined_source = owned(owner_plan['lifecycle_module'])
    assert sha(combined_source) == COMBINED_SHA
    prior_combined = read(Path(owner_plan['lifecycle_path']) / 'plan.json')
    prior567 = prior_combined['children']['gpu567']
    source = GPU4_STAGE / 'rlt_returned_cycle.py'
    assert sha(source) == CHILD_SHA
    R = load('bell_prepare_gpu4', source)
    R.parent_complete = repaired_parent_complete
    short_prepare(R)
    output.mkdir(mode=0o700)
    children, summaries = {}, {}
    stage = output / (args.cycle_prefix + '-gpu4')
    base = owned(read(GPU4_STAGE / 'plan.json')['base_module'])
    R.prepare(stage, GPU4_STAGE, source, NUMERIC_OWNER, base, **scope)
    summaries['gpu4'] = pin_prepared(stage, R, scope)
    children['gpu4'] = dict(path=str(stage), module=str(stage / 'rlt_returned_cycle.py'))
    if args.groups == 'all':
        previous_stage, previous_module = owned(prior567['path']), owned(prior567['module'])
        assert sha(previous_module) == prior567['module_sha256'] == CHILD_SHA
        assert sha(previous_stage / 'plan.json') == prior567['plan_sha256']
        R567 = load('bell_prepare_gpu567', source)
        short_prepare(R567)
        stage = output / (args.cycle_prefix + '-gpu567')
        R567.prepare(stage, previous_stage, previous_module, FOUR_OWNER, base, **scope)
        summaries['gpu567'] = pin_prepared(stage, R567, scope)
        children['gpu567'] = dict(path=str(stage), module=str(stage / 'rlt_returned_cycle.py'))
        C = load('bell_prepare_combined', combined_source)
        C.prepare(output / 'cycle', children)
    result = dict(prepared_only=True, processes_stopped=False, processes_launched=False,
        physical_gpus=[4] if args.groups == 'gpu4' else [4, 5, 6, 7], children=children,
        combined_path=str(output / 'cycle') if args.groups == 'all' else None,
        scope=scope, preparer_sha256=sha(__file__),
        checkpoint_steps={key: {k: row['recovery']['checkpoint']['step'] for k, row in plan['runs'].items()}
                          for key, plan in summaries.items()},
        new_runs={k: row['new_run'] for plan in summaries.values() for k, row in plan['runs'].items()})
    save_new(output / 'prepared-cycles.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
