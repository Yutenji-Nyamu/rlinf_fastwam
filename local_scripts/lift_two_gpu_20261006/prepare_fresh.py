"""Reuse the completed two-GPU smoke and prepare a fresh 200-round lift run."""
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D = S / 'lift-two-gpu-b32-v1'
SMOKE = S / 'runs/lift-two-gpu-b32-v2'
PREV = S / 'runs/lift-two-gpu-from0-v1'
F = S / 'lift-two-gpu-from0-v2'
O = S / 'runs/lift-two-gpu-from0-v2'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, text):
    p = Path(p)
    assert not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    p.write_text(text)
    p.chmod(0o500 if p.suffix == '.py' else 0o600)


def save(p, value):
    write(p, json.dumps(value, indent=2) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    assert not O.exists() and not (F / 'ready.json').exists()
    previous = read(D / 'prepared-v2/owner-plan.json')
    final = read(PREV / 'final.json')
    assert final['error']['type'] == 'AssertionError'
    assert final['error']['error'] == 'Smoke failed; do not start next N: formal'
    assert "KeyError: '/home/nvme/team-data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-two-gpu-from0-v1/code/fresh_owner.py'" in (PREV / 'formal/driver.log').read_text()
    assert final['recovery_error'] is None and final['rlt_return_dispatched']
    assert read(SMOKE / 'startup_smoke/result.json')['exit_code'] == 0
    assert read(SMOKE / 'startup-smoke-accepted.json')['passed']
    for path, digest in previous['source_sha256'].items():
        assert sha(path) == digest, path
    # Change only the optional compute-memory statistic; ownership and GPU
    # placement enumeration remain outside the timeout handler.
    text = Path(previous['base_owner_module']).read_text()
    before = """    value['compute_memory_csv'] = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader'],
        text=True, timeout=20).strip()
"""
    after = """    try:
        value['compute_memory_csv'] = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader'],
            text=True, timeout=20).strip()
    except subprocess.TimeoutExpired:
        value['compute_memory_csv'] = None
        value['compute_memory_error'] = {'type': 'TimeoutExpired', 'timeout_seconds': 20,
                                         'optional_statistic': True}
"""
    assert text.count(before) == 1
    text = text.replace(before, after)
    ast.parse(text)
    base = F / 'generated/opendw_owner_base.py'
    write(base, text)
    test = subprocess.run([previous['python'], '-B', str(F / 'code/test_resource_timeout.py'), str(base)],
                          text=True, capture_output=True, timeout=60)
    assert test.returncode == 0, test.stdout + test.stderr
    save(F / 'resource-timeout-cpu-test.json', json.loads(test.stdout))

    # The previous two-card owner has fully returned its exact RLT drivers.
    # Reborrow those identities, keeping GPU6/7 outside this lifecycle.
    parent_plan = read(PREV / 'owner-plan.json')
    oldcycle = read(Path(parent_plan['lifecycle_path']) / 'plan.json')
    assert oldcycle['physical_gpus'] == [4, 5]
    children = {}
    for key, name in [('gpu4', 'fresh2-g4'), ('gpu567', 'fresh2-g5')]:
        prior = oldcycle['children'][key]
        prior_plan = read(Path(prior['path']) / 'plan.json')
        source = D / 'generated' / key / 'rlt_returned_cycle.py'
        text = source.read_text()
        anchor = "release['gpus'] == [4, 5, 6, 7]"
        assert text.count(anchor) == 2
        # Only parent_complete is used; leave the unused adoption path intact.
        text = text.replace(anchor, "release['gpus'] == [4, 5]", 1)
        ast.parse(text)
        source = F / 'generated' / key / 'rlt_returned_cycle.py'
        write(source, text)
        module = load('fresh_cycle_' + key, source)
        target = F / 'cycles' / name
        module.prepare(target, Path(prior['path']), Path(prior['module']), PREV,
                       Path(prior_plan['base_module']), scope_activation=prior_plan['scope_activation'],
                       scope_manifest=prior_plan['scope_manifest'], scope_id=prior_plan['scope_id'])
        children[key] = {'path': str(target), 'module': str(target / 'rlt_returned_cycle.py')}
    combined_path = F / 'generated/rlt_returned_multigpu_cycle.py'
    write(combined_path, (D / 'generated/rlt_returned_multigpu_cycle.py').read_text())
    combined = load('fresh_cycle_combined', combined_path)
    cycle = F / 'cycles/cycle'
    combined.prepare(cycle, children)

    reference = D / 'prepared-v2/formal.yaml'
    cfg = read(reference)
    cfg = json.loads(json.dumps(cfg).replace(str(SMOKE / 'formal'), str(O / 'formal')))
    cfg['runner']['resume_dir'] = None
    cfg['actor']['fsdp_config']['resume_source_world_size'] = 1
    cfg['runner']['logger']['experiment_name'] = 'lift-two-gpu-from0-v2-formal'
    for component in ['actor', 'env', 'rollout']:
        cfg[component]['group_name'] = 'LiftFresh2_' + component
    config_path = F / 'prepared/formal.yaml'
    save(config_path, cfg)
    service = copy.deepcopy(previous['services'][0])
    service['argv'] = [v.replace(str(SMOKE), str(O)) for v in service['argv']]
    plan = copy.deepcopy(previous)
    plan.update(mode='two_gpu_b32_from0', owner_dir=str(O), lifecycle_path=str(cycle),
                lifecycle_module=str(cycle / 'rlt_returned_multigpu_cycle.py'),
                base_owner_module=str(base), reference_config=str(reference), resume_dir=None, resume_step=0,
                services=[service], reused_smoke_owner=str(SMOKE), retry_of=str(PREV),
                smoke_result=str(SMOKE / 'startup_smoke/result.json'),
                smoke_accepted=str(SMOKE / 'startup-smoke-accepted.json'))
    plan['trials'] = [{'key': 'formal', 'config': str(config_path), 'namespace': 'opendw_lift_fresh2_200_1007',
                       'episode_steps': 384, 'num_envs': 64, 'timeout_seconds': 60 * 86400}]
    files = list((F / 'code').glob('*.py')) + list((F / 'generated').rglob('*.py'))
    files += list((F / 'cycles').rglob('*.py')) + list((F / 'cycles').rglob('plan.json'))
    files += [reference, config_path, SMOKE / 'startup_smoke/result.json', SMOKE / 'startup-smoke-accepted.json']
    plan['source_sha256'].update({str(p): sha(p) for p in files})
    plan_path = F / 'prepared/owner-plan.json'
    save(plan_path, plan)
    # Match the driver's canonical argv spelling, not just preparation's alias.
    wrapper = load('fresh_owner_validate', (F / 'code/fresh_owner.py').resolve())
    module = wrapper.install(plan)
    module.validate(plan)
    ready = {'time': datetime.datetime.now().astimezone().isoformat(), 'plan': str(plan_path),
             'plan_sha256': sha(plan_path), 'entrypoint': str(F / 'code/fresh_owner.py'),
             'cpu_validate_passed': True, 'physical_gpus': [4, 5], 'resume_step': 0,
             'resume_dir': None, 'max_steps': 200, 'reused_smoke': str(SMOKE),
             'canonical_entrypoint_validated': True,
             'resource_timeout_cpu_test': read(F / 'resource-timeout-cpu-test.json')}
    save(F / 'ready.json', ready)
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    prepare()
