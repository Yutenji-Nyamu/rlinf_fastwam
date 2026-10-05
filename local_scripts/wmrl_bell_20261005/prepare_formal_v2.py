"""CPU-only v2 retry: clone the v1 plan, fix its native launcher, reborrow later.

Upload this script, prepare_cycles_v2.py and the fixed run_native_eval.py into
click-bell-v2/code/{ops,ops,binary} before invoking. No GPU work or signals.
"""
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D1, D2 = S / 'click-bell-v1', S / 'click-bell-v2'
O1, O2 = S / 'runs/click-bell-v1', S / 'runs/click-bell-v2'
P2 = D2 / 'prepared'
R = S / 'rlinf-opendw-bell-v1'


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
    assert path.resolve().is_relative_to(D2.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    path.chmod(0o600)


def replace_paths(value):
    if isinstance(value, dict):
        return {replace_paths(key): replace_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item) for item in value]
    if not isinstance(value, str):
        return value
    for old, new in [(D1 / 'code', D2 / 'code'), (D1 / 'generated', D2 / 'generated'),
                     (D1 / 'prepared', D2 / 'prepared'), (O1, O2)]:
        value = value.replace(str(old), str(new))
    return value.replace('127.0.0.1:18956', '127.0.0.1:18966').replace('127.0.0.1:18957', '127.0.0.1:18967')


def new_name(value):
    return value.replace('-v1-', '-v2-').replace('-v1', '-v2').replace('_v1', '_v2')


def config_names(cfg):
    for section in ('env', 'actor', 'rollout'):
        if isinstance(cfg.get(section), dict) and 'group_name' in cfg[section]:
            cfg[section]['group_name'] = new_name(cfg[section]['group_name'])
    cfg['runner']['logger']['experiment_name'] = new_name(cfg['runner']['logger']['experiment_name'])
    return cfg


def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not O2.exists() and not P2.exists() and not (D2 / 'generated').exists()
    fixed_launcher = D2 / 'code/binary/run_native_eval.py'
    cycles_script = D2 / 'code/ops/prepare_cycles_v2.py'
    assert fixed_launcher.is_file() and cycles_script.is_file()
    fixed_bytes = fixed_launcher.read_bytes()
    fixed_text = fixed_bytes.decode()
    assert 'fragment["CLUSTER_NAMESPACE"] = args.namespace' in fixed_text
    assert 'Cluster.NAMESPACE = args.namespace' in fixed_text
    assert 'env.stop().wait()' not in fixed_text and 'rollout.stop().wait()' not in fixed_text
    compile(fixed_text, str(fixed_launcher), 'exec')
    old = read(D1 / 'prepared/owner-plan.json')
    final = read(O1 / 'final.json')
    assert final['terminal_status'] == 'failed' and final['recovery_error'] is None
    assert final['rlt_borrowed'] is True and final['rlt_return_dispatched'] is True
    assert read(O1 / 'cleanup.json')['all_stopped'] is True
    assert old['repo'] == str(R)
    # The root staged exactly the repaired launcher in the new code directory;
    # retain these bytes while copying all other v1 implementation unchanged.
    original_code_digests = {str(path): sha(path) for path in (D1 / 'code').rglob('*.py')}
    shutil.copytree(D1 / 'code', D2 / 'code', dirs_exist_ok=True)
    fixed_launcher.write_bytes(fixed_bytes)
    shutil.copytree(D1 / 'generated', D2 / 'generated')
    assert all(sha(path) == digest for path, digest in original_code_digests.items())
    P2.mkdir(mode=0o700)
    # Preserve scope/environment bytes exactly when values do not change.
    for name in ['environment.json', 'graphics-fragment.json', 'native-environment.json', 'click-bell-seeds.json']:
        source, target = D1 / 'prepared' / name, P2 / name
        value = read(source)
        updated = replace_paths(value)
        if updated == value:
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
            target.chmod(0o600)
        else:
            save(target, updated)
    for name in ('formal.yaml', 'startup_smoke.yaml', 'native_rynn32.json', 'native_bell32.json'):
        save(P2 / name, config_names(replace_paths(read(D1 / 'prepared' / name))))
    assert sha(P2 / 'graphics-fragment.json') == sha(D1 / 'prepared/graphics-fragment.json')
    for name in ('formal.yaml', 'startup_smoke.yaml'):
        cfg = read(P2 / name)
        assert cfg['runner']['resume_dir'] is None and cfg['runner']['ckpt_path'] is None
        assert cfg['env']['train']['task_name'] == 'click_bell'
        assert cfg['env']['train']['initial_state_path'] == str(D1 / 'assets/click_bell_clean50_reset.npz')
    cycle_module = load('bell_v2_cycle_prepare', cycles_script)
    cycle_module.main()
    cycle_root = D2 / 'prepared-cycles'
    new_cycle = cycle_root / 'cycle'
    child_plan = read(new_cycle / 'plan.json')
    plan = replace_paths(copy.deepcopy(old))
    plan['lifecycle_path'] = str(new_cycle)
    plan['lifecycle_module'] = str(new_cycle / 'rlt_returned_multigpu_cycle.py')
    assert plan['owner_dir'] == str(O2) and plan['repo'] == str(R)
    for row in plan['trials']:
        row['namespace'] = new_name(row['namespace'])
        row['config_sha256'] = sha(row['config'])
    for row in plan['prechecks']:
        if row['kind'] == 'native':
            row['namespace'] = new_name(row['namespace'])
            argv = row['commands'][0]['argv']
            argv[argv.index('--namespace') + 1] = row['namespace']
    for service, port in zip(plan['services'], (18966, 18967)):
        argv = service['argv']
        argv[argv.index('--port') + 1] = str(port)
        assert service['url'] == 'http://127.0.0.1:' + str(port)
        # Keep downloaded weights/T5 and the already-working true-B16 service.
        assert argv[argv.index('--reward-checkpoint') + 1] == service['reward_checkpoint']
    plan['protocol_reference']['sha256'] = sha(plan['protocol_reference']['config'])
    plan['graphics_scope']['environment_fragment_sha256'] = sha(plan['graphics_scope']['environment_fragment_file'])
    plan['native_eval_seeds_sha256'] = sha(P2 / 'click-bell-seeds.json')
    old_cycle_root = Path(old['lifecycle_path']).parent
    files = set()
    for original in old['source_sha256']:
        if Path(original).is_relative_to(old_cycle_root):
            continue
        files.add(Path(replace_paths(original)))
    files.update([Path(__file__), cycles_script, fixed_launcher, new_cycle / 'plan.json',
                  Path(plan['lifecycle_module']), cycle_root / 'prepared-cycles.json'])
    for child in child_plan['children'].values():
        files.update([Path(child['path']) / 'plan.json', Path(child['module'])])
    plan['source_sha256'] = {str(path): sha(path) for path in sorted(files)}
    plan['retry'] = dict(parent_owner=str(O1), parent_plan_sha256=sha(O1 / 'owner-plan.json'),
        parent_final_sha256=sha(O1 / 'final.json'), fresh_rlt_cycle=str(new_cycle),
        implementation_change='native launcher propagates CLUSTER_NAMESPACE and Cluster.NAMESPACE; uses owner cleanup instead of nonexistent WorkerGroup.stop',
        old_launcher_sha256=sha(D1 / 'code/binary/run_native_eval.py'), new_launcher_sha256=sha(fixed_launcher),
        reused_policy_repo=str(R), reused_native_repo=str(S / 'rlinf-rynn-binary-v1'),
        sampling_budget_changed=False, new_gate_added=False, starts_from_original_sft=True)
    save(P2 / 'owner-plan.json', plan)
    entry = D2 / 'generated/owner/opendw_formal_owner.py'
    wrapper = load('bell_v2_preflight_formal', entry)
    module = wrapper.install(plan)
    prechecks = load('bell_v2_preflight_prechecks', plan['prechecks_module'])
    module = prechecks.install(module, plan)
    module.validate(plan)
    result = dict(time=datetime.datetime.now().astimezone().isoformat(), prepared=True, cpu_validate_passed=True,
        launched=False, processes_stopped=False, plan=str(P2 / 'owner-plan.json'), sha256=sha(P2 / 'owner-plan.json'),
        entrypoint=str(entry), owner_dir=str(O2), lifecycle_path=str(new_cycle),
        precheck_order=[row['key'] for row in plan['prechecks']], retry=plan['retry'])
    save(P2 / 'ready.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
