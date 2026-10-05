"""Prepare one fresh, normal reborrow after Rynn v1 returned RLT; no signals."""
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
OLD_D = S / 'rynn-control-v1'
PARENT = S / 'runs/rynn-success-v1'
D = S / 'rynn-control-v2'
CODE = D / 'code'
STATE = D / 'prepared'
OWNER = S / 'runs/rynn-success-v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
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


def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert not OWNER.exists() and not (STATE / 'plan.json').exists()
    tests = read(STATE / 'cpu-tests.json')
    assert tests['all_cpu_tests_passed'] is True
    for path, digest in tests['source_sha256'].items():
        assert sha(path) == digest, 'Tested repair source changed: ' + path
    old = read(PARENT / 'owner-plan.json')
    final = read(PARENT / 'final.json')
    assert final['terminal_status'] == 'failed' and final['recovery_error'] is None
    assert final['rlt_return_dispatched'] is True and final['rlt_borrowed'] is True
    assert read(PARENT / 'cleanup.json')['all_stopped'] is True
    # Old plans and code remain immutable, including the defective service.
    for path, digest in old['source_sha256'].items():
        assert sha(path) == digest, 'V1 frozen provenance changed: ' + path
    support = Path(old['base_formal_wrapper']).parent
    child_source = support / 'rlt_returned_cycle.py'
    combined_source = support / 'rlt_returned_multigpu_cycle.py'
    for path in (child_source, combined_source):
        assert old['source_sha256'][str(path)] == sha(path)
    R = load('rynn_v2_returned_child', child_source)
    C = load('rynn_v2_returned_combined', combined_source)
    prior_cycle = read(Path(old['lifecycle_path']) / 'plan.json')
    scope = copy.deepcopy(old['graphics_scope'])
    active = read(scope['activation_receipt'])
    assert active['status'] == 'active'
    base = Path(read(Path(prior_cycle['children']['gpu567']['path']) / 'plan.json')['base_module'])
    assert sha(base) == R.BASE_SHA
    children = {}
    for key in ('gpu4', 'gpu567'):
        prior = prior_cycle['children'][key]
        stage = STATE / ('rynn-v2-reborrow-' + key)
        R.prepare(stage, Path(prior['path']), Path(prior['module']), PARENT, base,
            scope_activation=scope['activation_receipt'], scope_manifest=active['manifest'], scope_id=active['scope_id'])
        children[key] = dict(path=str(stage), module=str(stage / 'rlt_returned_cycle.py'))
    cycle = STATE / 'cycle'
    C.prepare(cycle, children)
    environment_path = STATE / 'environment.json'
    save(environment_path, read(old['environment_file']))
    environment_path.chmod(0o600)
    services = copy.deepcopy(old['services'])
    for service in services:
        service['argv'] = [arg.replace(str(OLD_D / 'code'), str(CODE)).replace(str(PARENT), str(OWNER))
                           for arg in service['argv']]
        service['cwd'] = str(CODE)
    trials = []
    for row in old['trials']:
        cfg = read(row['config'])
        cfg = json.loads(json.dumps(cfg).replace(str(PARENT), str(OWNER))
            .replace('opendw-adjust-bottle-rynn-success-v1', 'opendw-adjust-bottle-rynn-success-v2')
            .replace('opendw-adjust-bottle-rynn-startup-v1', 'opendw-adjust-bottle-rynn-startup-v2'))
        cfg['env']['train']['rynn_run_id'] = OWNER.name + '/' + row['key']
        path = STATE / (row['key'] + '.yaml')
        save(path, cfg)
        updated = dict(row, config=str(path), config_sha256=sha(path), namespace=row['namespace'].removesuffix('_v1') + '_v2')
        trials.append(updated)
    source = dict(old['source_sha256'])
    source.update({str(path): sha(path) for path in CODE.glob('*.py')})
    source[str(cycle / 'rlt_returned_multigpu_cycle.py')] = sha(cycle / 'rlt_returned_multigpu_cycle.py')
    for child in children.values():
        stage = Path(child['path'])
        source.update(read(stage / 'plan.json')['frozen_files'])
        source[child['module']] = sha(child['module'])
        source[str(stage / 'plan.json')] = sha(stage / 'plan.json')
    gate = copy.deepcopy(old['rm_gate'])
    gate['argv'] = [arg.replace(str(OLD_D / 'code'), str(CODE)).replace(str(PARENT), str(OWNER)) for arg in gate['argv']]
    gate['cwd'] = str(CODE)
    gate['output'] = str(OWNER / 'rm_gate/result.json')
    plan = {key: copy.deepcopy(old[key]) for key in ('mode', 'start_mode', 'startup_smoke', 'wm_batch_size',
        'base_owner_module', 'base_formal_wrapper', 'python', 'repo', 'repo_head', 'physical_gpus',
        'restore_wait_seconds', 'native_eval_seeds_sha256', 'protocol_reference', 'resume_checkpoint', 'budget')}
    plan.update(owner_dir=str(OWNER), lifecycle_path=str(cycle), lifecycle_module=str(cycle / 'rlt_returned_multigpu_cycle.py'),
        environment_file=str(environment_path), source_sha256=source, services=services, trials=trials,
        graphics_scope=scope, rm_gate=gate,
        returned_reborrow=dict(parent_owner=str(PARENT), owner_plan_sha256=sha(PARENT / 'owner-plan.json'),
            final_sha256=sha(PARENT / 'final.json'), reason='Rynn v1 model dtype failure; tested BF16 transfer repair'))
    assert plan['repo'] == str(S / 'rlinf-rynn-v1') and plan['repo_head'] == '2151a08ee1bd75df1bef0d8190e594bd5c7f7977'
    assert str(CODE / 'rynn_formal_owner_v2.py') in source
    F = load('rynn_v2_owner_preflight', CODE / 'rynn_formal_owner_v2.py')
    module = F.install(plan)
    module.validate(plan)
    save(STATE / 'plan.json', plan)
    receipt = dict(time=datetime.datetime.now().astimezone().isoformat(), plan=str(STATE / 'plan.json'),
        plan_sha256=sha(STATE / 'plan.json'), owner_dir=str(OWNER), current_rlt_preserved=True,
        cpu_preflight_passed=True, parent_final_sha256=sha(PARENT / 'final.json'),
        checkpoint_steps={key: {run: row['recovery']['checkpoint']['step'] for run, row in
            read(Path(child['path']) / 'plan.json')['runs'].items()} for key, child in children.items()})
    save(D / 'prepared.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
