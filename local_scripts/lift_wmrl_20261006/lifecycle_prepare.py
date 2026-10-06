"""Prepare fresh four-GPU WMRL borrowing from two completed owner chains.

CPU preparation only. GPU4 currently belongs to the RLT returned by task RM v2;
GPU5/6/7 belong to the RLT returned by click-bell v2. Runtime stop/resume and
checkpoint selection are reused unchanged. The single-card parent's validator
is narrowly rebound to the completed task RM v2 receipt; new run names are short.
"""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys


ROOT = Path('/data/chenyiteng')
S = ROOT / 'projects/opendw-robotwin-smoke-20261003'
GPU4_PARENT = S / 'task-reward-v2/run'
GPU4_PREVIOUS = S / 'task-reward-v2/prepared/rm1006-v2'
GPU567_PARENT = S / 'runs/click-bell-v2'
SCOPE = S / 'rynn-numeric-v1/rlt-return-repair-v2/scope'
CHILD_SHA = '2667484807da8065ce523ea398b5502b384e39898de58b98d50aef72a71ab7b3'
COMBINED_SHA = '6d92df58de2f0d905273eedc879cc3cdc3d1523d58d5db57f4403e748fb961a1'
NAMED_SCOPE_SHA = '96d183e00506cd44345cfadcd61997f10ef8b4d810b9a2aee518ef2930b70114'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned(path, exists=True):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
    assert not path.is_symlink()
    if exists:
        assert path.exists() and path.stat().st_uid == 20001
    return path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o600)


def rebind_single_parent(source):
    """Change one parent path inside the already-proven single-card validator."""
    parsed = ast.parse(source)
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'parent_complete']
    assert len(nodes) == 1
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    function = ''.join(lines[node.lineno - 1:node.end_lineno])
    prior_parent = str(S / 'task-reward-v1/run')
    assert function.count(prior_parent) == 1
    for marker in ["'task-rm-single-gpu'", "'rm-release.json'", "'untouched_gpus'",
                   "'dojo-release-verified.json'", "'rlt-return-dispatched.json'"]:
        assert marker in function, 'Unexpected prior parent validator: ' + marker
    replacement = function.replace(prior_parent, str(GPU4_PARENT), 1)
    revised = ''.join(lines[:node.lineno - 1]) + replacement + ''.join(lines[node.end_lineno:])
    after = ast.parse(revised)
    old_other = [ast.dump(n, include_attributes=False) for n in parsed.body if n is not node]
    new_other = [ast.dump(n, include_attributes=False) for n in after.body
                 if not (isinstance(n, ast.FunctionDef) and n.name == 'parent_complete')]
    assert old_other == new_other, 'Changes outside parent_complete are forbidden'
    return revised


def short_prepare(module):
    source = inspect.getsource(module.prepare)
    before = "        new_run = original.with_name(original.name + '-after-' + stage.name)"
    after = ("        new_run = Path('/data/chenyiteng/results/rlinf-rlt') / "
             "('rlt-sz3-g' + str(gpu) + '-after-' + stage.name)\n"
             "        assert len(new_run.name.encode()) < 100")
    assert source.count(before) == 1
    exec(compile(source.replace(before, after, 1), str(__file__) + ':short_prepare', 'exec'), module.__dict__)


def named_scope():
    activation, manifest = owned(SCOPE / 'activation.json'), owned(SCOPE / 'scope.json')
    active, scope = read(activation), read(manifest)
    assert active['status'] == 'active' and active['uid'] == 20001
    assert active['manifest'] == str(manifest) and active['manifest_sha256'] == sha(manifest)
    assert scope['physical_gpus'] == [4, 5, 6, 7] and scope['cpu_full_mask_target'] == 4
    assert active['scope_id'] == scope['token']
    runtime = owned(active['runtime_path'])
    assert sha(runtime) == active['runtime_sha256'] == NAMED_SCOPE_SHA
    load('lift_named_scope', runtime).read_manifest(manifest)
    fragment = owned(active['environment_fragment_file'])
    assert sha(fragment) == active['environment_fragment_sha256']
    assert read(fragment) == active['environment_fragment']
    assert read(fragment)['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'] == str(manifest)
    return dict(scope_activation=str(activation), scope_manifest=str(manifest), scope_id=active['scope_id']), [activation, manifest, runtime, fragment]


def prepare(output):
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Prepare with CUDA hidden'
    output = owned(output, False)
    assert output.resolve().is_relative_to(S.resolve()) and not output.exists()
    owned(output.parent)
    scope, scope_files = named_scope()
    old4 = read(owned(GPU4_PARENT / 'owner-plan.json'))
    old567 = read(owned(GPU567_PARENT / 'owner-plan.json'))
    assert old4['lifecycle_path'] == str(GPU4_PREVIOUS)
    source4 = owned(old4['lifecycle_module'])
    assert source4 == GPU4_PREVIOUS / 'rlt_returned_cycle.py'
    assert sha(source4) == old4['source_sha256'][str(source4)]
    assert sha(GPU4_PREVIOUS / 'plan.json') == old4['source_sha256'][str(GPU4_PREVIOUS / 'plan.json')]
    assert read(GPU4_PARENT / 'final.json')['terminal_status'] == 'completed'
    combined_source = owned(old567['lifecycle_module'])
    assert sha(combined_source) == old567['source_sha256'][str(combined_source)] == COMBINED_SHA
    combined_previous = owned(old567['lifecycle_path'])
    assert sha(combined_previous / 'plan.json') == old567['lifecycle_plan_sha256']
    children = read(combined_previous / 'plan.json')['children']
    assert set(children) == {'gpu4', 'gpu567'}
    previous567, source567 = owned(children['gpu567']['path']), owned(children['gpu567']['module'])
    assert previous567 == S / 'click-bell-v2/prepared-cycles/bell-v2-gpu567'
    assert sha(source567) == children['gpu567']['module_sha256'] == CHILD_SHA
    assert sha(previous567 / 'plan.json') == children['gpu567']['plan_sha256']
    prior_plans = {'gpu4': read(GPU4_PREVIOUS / 'plan.json'), 'gpu567': read(previous567 / 'plan.json')}
    assert all(p['scope_manifest'] == scope['scope_manifest'] for p in prior_plans.values())
    original_paths = [GPU4_PREVIOUS / 'plan.json', source4, previous567 / 'plan.json', source567,
                      GPU4_PARENT / 'owner-plan.json', GPU4_PARENT / 'final.json',
                      GPU567_PARENT / 'owner-plan.json', GPU567_PARENT / 'final.json',
                      combined_previous / 'plan.json', combined_source]
    original_sha = {str(p): sha(p) for p in original_paths}
    # Validate GPU567's completed owner before creating any preparation output.
    R567 = load('lift_prepare_gpu567', source567)
    assert R567.parent_complete(GPU567_PARENT, previous567, source567)[0] == 'gpu567'
    revised = rebind_single_parent(source4.read_text())
    output.mkdir(mode=0o700)
    adapter_dir = output / 'single-parent-adapter'
    adapter_dir.mkdir(mode=0o700)
    adapter = adapter_dir / 'rlt_returned_cycle.py'
    with adapter.open('x') as stream:
        stream.write(revised)
    adapter.chmod(0o500)
    delta = adapter_dir / 'source-delta.json'
    save(delta, dict(original=str(source4), original_sha256=sha(source4), adapted=str(adapter),
                    adapted_sha256=sha(adapter), completed_parent=str(GPU4_PARENT),
                    change='Only parent_complete fixed owner path from task-reward-v1/run to task-reward-v2/run',
                    other_top_level_code_unchanged=True, stop_resume_load_plan_unchanged=True))
    R4 = load('lift_prepare_gpu4', adapter)
    assert R4.parent_complete(GPU4_PARENT, GPU4_PREVIOUS, source4)[0] == 'gpu4'
    donors = {'gpu4': (R4, GPU4_PREVIOUS, source4, GPU4_PARENT, 'lift-g4-v1'),
              'gpu567': (R567, previous567, source567, GPU567_PARENT, 'lift-g567-v1')}
    fresh_children, summaries = {}, {}
    for key in ('gpu4', 'gpu567'):
        module, previous, source, parent, name = donors[key]
        short_prepare(module)
        stage = output / name
        module.prepare(stage, previous, source, parent, owned(prior_plans[key]['base_module']), **scope)
        new = read(stage / 'plan.json')
        assert new['group'] == key and new['completed_owner'] == str(parent)
        assert not any((stage / p).exists() for p in ('rlt-stopped.json', 'old-stop-attempt.json', 'resumed-dispatched.json'))
        extras = [owned(__file__), *scope_files, adapter, delta]
        new['frozen_files'].update({str(p): sha(p) for p in extras})
        new['prepare_only_changes'] = dict(short_run_names=True, parent_owner=str(parent),
            original_runtime_module_sha256=sha(source), runtime_module_sha256=sha(stage / 'rlt_returned_cycle.py'),
            named_graphics_scope=scope['scope_manifest'], stop_or_start_performed=False,
            parent_validator_rebound=(key == 'gpu4'))
        temporary = stage / 'plan.prepare-final.tmp'
        save(temporary, new)
        os.replace(temporary, stage / 'plan.json')
        module.load_plan(stage)
        assert sha(stage / 'rlt_returned_cycle.py') == sha(adapter if key == 'gpu4' else source567)
        fresh_children[key] = dict(path=str(stage), module=str(stage / 'rlt_returned_cycle.py'))
        summaries[key] = new
    combined = load('lift_prepare_combined', combined_source)
    combined.prepare(output / 'cycle', fresh_children)
    assert all(sha(p) == value for p, value in original_sha.items()), 'Original owner or lifecycle changed'
    result = dict(time=datetime.datetime.now().astimezone().isoformat(), prepared_only=True,
        processes_stopped=False, processes_launched=False, physical_gpus=[4, 5, 6, 7],
        parents={'gpu4': str(GPU4_PARENT), 'gpu567': str(GPU567_PARENT)}, scope=scope,
        combined_path=str(output / 'cycle'), children=fresh_children,
        checkpoint_steps={key: {role: row['recovery']['checkpoint']['step'] for role, row in p['runs'].items()}
                          for key, p in summaries.items()},
        new_runs={role: row['new_run'] for p in summaries.values() for role, row in p['runs'].items()},
        original_lifecycle_files_preserved=True, preparer_sha256=sha(__file__))
    save(output / 'prepared-cycles.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.prepared_dir)


if __name__ == '__main__':
    main()
