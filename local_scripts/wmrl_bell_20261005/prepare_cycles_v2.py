"""Prepare new RLT cycles after click-bell-v1's verified normal return.

CPU only. No stop/start, no adoption, no old plan edits. The runtime lifecycle
is copied byte-for-byte; only preparation chooses bounded new run names.
"""
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D = S / 'click-bell-v2'
OUTPUT = D / 'prepared-cycles'
PARENT = S / 'runs/click-bell-v1'
SCOPE = S / 'rynn-numeric-v1/rlt-return-repair-v2/scope'
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


def save_new(path, value):
    assert path.resolve().is_relative_to(OUTPUT.resolve())
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def short_prepare(module):
    source = inspect.getsource(module.prepare)
    old = "        new_run = original.with_name(original.name + '-after-' + stage.name)"
    new = ("        new_run = Path('/data/chenyiteng/results/rlinf-rlt') / "
           "('rlt-sz3-g' + str(gpu) + '-after-' + stage.name)\n"
           "        assert len(new_run.name.encode()) < 100")
    assert source.count(old) == 1
    exec(compile(source.replace(old, new, 1), str(Path(__file__)) + ':short_prepare', 'exec'), module.__dict__)


def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use CPU preparation with CUDA hidden'
    assert not OUTPUT.exists(), 'Fresh v2 cycle directory required'
    assert not (S / 'runs/click-bell-v2').exists(), 'Prepare before creating the v2 owner'
    old = read(PARENT / 'owner-plan.json')
    final = read(PARENT / 'final.json')
    assert old['owner_dir'] == str(PARENT) and old['mode'] == 'multigpu_formal'
    assert final['terminal_status'] == 'failed' and final['recovery_error'] is None
    assert final['rlt_borrowed'] is True and final['rlt_return_dispatched'] is True
    assert read(PARENT / 'cleanup.json')['all_stopped'] is True
    previous_cycle = Path(old['lifecycle_path'])
    combined_source = Path(old['lifecycle_module'])
    assert sha(combined_source) == COMBINED_SHA == old['source_sha256'][str(combined_source)]
    combined_plan = read(previous_cycle / 'plan.json')
    assert set(combined_plan['children']) == {'gpu4', 'gpu567'}
    assert sha(previous_cycle / 'plan.json') == old['lifecycle_plan_sha256']

    manifest_path = SCOPE / 'scope.json'
    active_path = SCOPE / 'activation.json'
    active, manifest = read(active_path), read(manifest_path)
    assert active['status'] == 'active' and active['uid'] == 20001
    assert active['manifest'] == str(manifest_path) and active['manifest_sha256'] == sha(manifest_path)
    assert manifest['physical_gpus'] == [4, 5, 6, 7] and manifest['cpu_full_mask_target'] == 4
    runtime_path = Path(active['runtime_path'])
    assert sha(runtime_path) == active['runtime_sha256'] == NAMED_SCOPE_SHA
    runtime = load('bell_v2_named_scope_readonly', runtime_path)
    runtime.read_manifest(manifest_path)
    fragment_path = Path(active['environment_fragment_file'])
    assert sha(fragment_path) == active['environment_fragment_sha256']
    assert read(fragment_path) == active['environment_fragment']
    scope = dict(scope_activation=str(active_path), scope_manifest=str(manifest_path), scope_id=active['scope_id'])

    # Check both parents before writing anything. R.prepare performs these
    # checks again with live PID/namespace/checkpoint validation before copying.
    donors, original_digests = {}, {}
    for key, child in combined_plan['children'].items():
        previous, source = Path(child['path']), Path(child['module'])
        assert sha(source) == child['module_sha256'] == CHILD_SHA
        assert sha(previous / 'plan.json') == child['plan_sha256']
        module = load('bell_v2_prepare_' + key, source)
        group, _, _ = module.parent_complete(PARENT, previous, source)
        assert group == key
        previous_plan = read(previous / 'plan.json')
        assert previous_plan['scope_manifest'] == str(manifest_path)
        base = Path(previous_plan['base_module'])
        donors[key] = (previous, source, base, module)
        original_digests[str(previous / 'plan.json')] = sha(previous / 'plan.json')
        original_digests[str(source)] = sha(source)
    D.mkdir(mode=0o700, exist_ok=True)
    OUTPUT.mkdir(mode=0o700)
    children, summaries = {}, {}
    for key in ('gpu4', 'gpu567'):
        previous, source, base, module = donors[key]
        short_prepare(module)
        stage = OUTPUT / ('bell-v2-' + key)
        module.prepare(stage, previous, source, PARENT, base, **scope)
        fresh = read(stage / 'plan.json')
        assert not any((stage / name).exists() for name in ('old-stop-attempt.json', 'rlt-stopped.json', 'resumed-dispatched.json'))
        extras = [Path(__file__), active_path, manifest_path, runtime_path, fragment_path]
        fresh['frozen_files'].update({str(path): sha(path) for path in extras})
        fresh['prepare_only_changes'] = dict(short_run_names=True, original_runtime_module_sha256=CHILD_SHA,
            named_graphics_scope=str(manifest_path), parent_owner=str(PARENT), stop_or_start_performed=False)
        temporary = stage / 'plan.prepare-final.tmp'
        save_new(temporary, fresh)
        os.replace(temporary, stage / 'plan.json')
        module.load_plan(stage)
        assert sha(stage / 'rlt_returned_cycle.py') == CHILD_SHA
        children[key] = dict(path=str(stage), module=str(stage / 'rlt_returned_cycle.py'))
        summaries[key] = fresh
    combined = load('bell_v2_prepare_combined', combined_source)
    combined.prepare(OUTPUT / 'cycle', children)
    assert all(sha(path) == value for path, value in original_digests.items()), 'Original lifecycle changed'
    result = dict(time=datetime.datetime.now().astimezone().isoformat(), prepared_only=True,
        processes_stopped=False, processes_launched=False, parent_owner=str(PARENT),
        parent_final_sha256=sha(PARENT / 'final.json'), physical_gpus=[4, 5, 6, 7],
        combined_path=str(OUTPUT / 'cycle'), children=children, scope=scope,
        checkpoint_steps={key: {role: row['recovery']['checkpoint']['step'] for role, row in value['runs'].items()}
                          for key, value in summaries.items()},
        new_runs={role: row['new_run'] for value in summaries.values() for role, row in value['runs'].items()},
        original_cycle_files_preserved=True, preparer_sha256=sha(__file__))
    save_new(OUTPUT / 'prepared-cycles.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
