"""CPU assemble lift_pot baseline -> full-parallel smoke -> formal WMRL.

Run after reset/recipe/lifecycle preparation. This reuses the original owner,
native phase runner, cleanup and RLT return; it does not start or stop anything.
The supplied RLinf checkout is read-only and may be the proven bell checkout
when its generic environment already supports the new task and reward source.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/data/chenyiteng')
S = ROOT / 'projects/opendw-robotwin-smoke-20261003'
D = S / 'lift-pot-v1'
O = S / 'runs/lift-pot-v1'
DONOR = S / 'click-bell-v2'
NATIVE = S / 'rlinf-rynn-binary-v1'
RM = S / 'task-reward-v2'
SCOPE = S / 'rynn-numeric-v1/rlt-return-repair-v2/scope'
HEAD = '2151a08ee1bd75df1bef0d8190e594bd5c7f7977'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def owned(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
    assert not path.is_symlink() and path.stat().st_uid == 20001
    return path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    assert Path(path).resolve().is_relative_to(D.resolve())
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    Path(path).chmod(0o600)


def copy_new(source, target):
    owned(source)
    assert target.resolve().is_relative_to(D.resolve()) and not target.exists()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(Path(source).read_bytes())
    target.chmod(0o500 if target.suffix == '.py' else 0o600)
    assert sha(source) == sha(target)


def remove_old_hook(text):
    before = """    if args.action == 'owner':
        hook_path = Path(plan['prechecks_module'])
        assert sha(hook_path) == plan['source_sha256'][str(hook_path)]
        hook_spec = importlib.util.spec_from_file_location('bell_finite_prechecks', hook_path)
        hook = importlib.util.module_from_spec(hook_spec)
        hook_spec.loader.exec_module(hook)
        module = hook.install(module, plan)
        module.owner_main(plan)"""
    after = before.replace("'bell_finite_prechecks'", "'lift_native_baseline_precheck'")
    assert text.count(before) == 1
    return text.replace(before, after, 1)


def native_config(prepared, formal):
    """Keep the recipe's proven N16x2 capture; only route its output phase."""
    cfg = json.loads(json.dumps(prepared).replace(str(O / 'native_baseline32'), str(O / 'native_lift32')))
    e = cfg['env']['eval']
    assert e['task_config']['task_name'] == 'lift_pot' and 'train' not in cfg['env']
    assert e['total_num_envs'] == 16 and e['rollout_epoch'] == 2
    assert e['max_episode_steps'] == e['max_steps_per_rollout_epoch'] == e['task_config']['step_lim'] == 384
    assert e['use_fixed_reset_state_ids'] is False
    assert e['seeds_path'] == formal['env']['eval']['seeds_path']
    assert cfg['rollout']['model'] == formal['rollout']['model']
    assert cfg['cluster']['component_placement'] == {'env': '4', 'rollout': '4'}
    assert cfg['runner']['only_eval'] and cfg['runner']['ckpt_path'] is None
    assert cfg['runner']['logger']['log_path'] == str(O / 'native_lift32')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe-dir', type=Path, default=D / 'prepared')
    parser.add_argument('--private-repo', type=Path, required=True)
    parser.add_argument('--code-dir', type=Path, default=D / 'code')
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU assembly requires hidden CUDA'
    p, c, repo = owned(args.recipe_dir), owned(args.code_dir), owned(args.private_repo)
    assert p == D / 'prepared' and c.resolve().is_relative_to(D.resolve())
    assert not O.exists() and not (p / 'owner-plan.json').exists() and not (D / 'generated').exists()
    old = read(DONOR / 'prepared/owner-plan.json')
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == HEAD == old['repo_head']
    assert subprocess.check_output(['git', '-C', str(NATIVE), 'rev-parse', 'HEAD'], text=True).strip() == HEAD
    formal, smoke = read(p / 'formal.yaml'), read(p / 'startup_smoke.yaml')
    contract = read(p / 'config-contract.json')
    assert formal['env']['train']['task_name'] == 'lift_pot'
    assert contract['reward_checkpoint_sha256'] == sha(contract['reward_checkpoint'])
    assert contract['native_seeds_sha256'] == sha(formal['env']['eval']['seeds_path'])
    assert formal['runner']['logger']['log_path'] == str(O / 'formal')
    assert smoke['runner']['logger']['log_path'] == str(O / 'startup_smoke')
    urls = formal['env']['train']['service_urls']
    assert urls == ['http://127.0.0.1:18976', 'http://127.0.0.1:18977']
    assert formal['runner']['save_interval'] == formal['runner']['val_check_interval'] == 10
    cycle = owned(D / 'prepared-cycles/cycle')
    lifecycle = owned(cycle / 'rlt_returned_multigpu_cycle.py')
    cp = read(cycle / 'plan.json')
    assert cp['physical_gpus'] == [4, 5, 6, 7] and sha(lifecycle) == cp['script_sha256']
    assert not (cycle / 'rlt-stopped.json').exists()
    patch_path = owned(c / 'patch_lift.py')
    patch = load('lift_task_source_transform', patch_path)
    g = D / 'generated'
    g.mkdir(mode=0o700)
    service_donor = Path(old['services'][0]['argv'][3])
    source_rows = [(service_donor, g / 'service/opendw_service_batched.py', patch.service_source),
        (Path(old['base_owner_module']), g / 'owner/opendw_multigpu_owner.py', patch.base_owner_source),
        (DONOR / 'generated/owner/opendw_formal_owner.py', g / 'owner/opendw_formal_owner.py',
         lambda text: remove_old_hook(patch.formal_owner_source(text)))]
    generated = []
    for source, target, transform in source_rows:
        assert sha(owned(source)) == old['source_sha256'][str(source)]
        text = transform(source.read_text())
        compile(text, str(target), 'exec')
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with target.open('x') as stream:
            stream.write(text)
        target.chmod(0o500)
        generated.append(dict(donor=str(source), donor_sha256=sha(source), target=str(target), sha256=sha(target)))
    # The service runs in OpenDW's venv; keep its proven batch implementation and
    # action telemetry byte-identical, and load the exact training architecture.
    copied = []
    for source, target in [(service_donor.parent / name, g / 'service' / name)
                          for name in ('wm_batch.py', 'opendw_action_telemetry.py')]:
        assert sha(source) == old['source_sha256'][str(source)]
        copy_new(source, target)
        copied.append(dict(source=str(source), target=str(target), sha256=sha(target)))
    for source, target in [(c / 'rm_adapter.py', g / 'service/rm_adapter.py'),
                           (RM / 'code/rm_inference.py', g / 'service/rm_inference.py')]:
        copy_new(source, target)
        copied.append(dict(source=str(source), target=str(target), sha256=sha(target)))
    rm_owner = read(RM / 'run/owner-plan.json')
    assert sha(RM / 'code/rm_inference.py') == rm_owner['source_sha256'][str(RM / 'code/rm_inference.py')]
    save(p / 'generated-sources.json', dict(generated=generated, copied=copied))
    assert sha(old['environment_file']) == old['source_sha256'][old['environment_file']]
    old_env = read(old['environment_file'])
    environment = {key: str(value).replace(old['repo'], str(repo)) for key, value in old_env.items()}
    environment.update(RLINF_CODE_WORKING_DIR=str(repo), HOME='/home/chenyiteng', USER='chenyiteng',
                       LOGNAME='chenyiteng', PYTHONDONTWRITEBYTECODE='1')
    assert not any(k in environment for k in ('CUDA_VISIBLE_DEVICES', 'LD_PRELOAD', 'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'))
    save(p / 'environment.json', environment)
    assert sha(old['graphics_scope']['environment_fragment_file']) == old['graphics_scope']['environment_fragment_sha256']
    fragment = read(old['graphics_scope']['environment_fragment_file'])
    fragment['PYTHONPATH'] = fragment['PYTHONPATH'].replace(old['repo'], str(repo))
    assert str(repo) in fragment['PYTHONPATH'].split(':')
    save(p / 'graphics-fragment.json', fragment)
    native_environment = {key: str(value).replace(str(repo), str(NATIVE))
                          for key, value in dict(environment, **fragment).items()}
    native_environment['RLINF_CODE_WORKING_DIR'] = str(NATIVE)
    assert str(NATIVE) in native_environment['PYTHONPATH'].split(':')
    assert 'native_binary_recorder' in (NATIVE / 'rlinf/envs/robotwin/robotwin_env.py').read_text()
    save(p / 'native-environment.json', native_environment)
    cfg = native_config(read(p / 'native_baseline.json'), formal)
    save(p / 'native_lift32.json', cfg)
    seeds_path = Path(cfg['env']['eval']['seeds_path'])
    seeds = read(seeds_path)['lift_pot']['success_seeds']
    excluded = read(RM / 'native-lift128/native_seeds.json')['lift_pot']['success_seeds']
    assert len(seeds) == len(set(seeds)) == 32 and not set(seeds).intersection(excluded)
    save(p / 'native-capture-plan.json', dict(task='lift_pot', native_episodes=32,
        selected_seed_ids=seeds, action_chunk=32, horizon=384, original_sft=True,
        rm_collection_seed_overlap=0, seeds_sha256=sha(seeds_path)))
    native_launcher = owned(DONOR / 'code/binary/run_native_eval.py')
    native_validator = owned(RM / 'code/validate_capture.py')
    assert sha(native_launcher) == old['source_sha256'][str(native_launcher)]
    assert sha(native_validator) == rm_owner['source_sha256'][str(native_validator)]
    target = O / 'native_lift32'
    namespace = 'opendw_sz3_lift_native32_v1'
    commands = [dict(argv=[cp['python'], '-u', '-B', str(native_launcher), '--config', str(p / 'native_lift32.json'),
        '--receipt-dir', str(target), '--private-repo', str(NATIVE), '--environment-fragment', str(p / 'native-environment.json'),
        '--capture-dir', str(target / 'capture'), '--capture-mode', 'reward_native', '--namespace', namespace,
        '--ray-address', cp['ray_address']], cwd=str(NATIVE), mode='ray', environment_file=str(p / 'native-environment.json')),
        dict(argv=[cp['python'], '-u', '-B', str(native_validator), '--capture-dir', str(target / 'capture'),
            '--plan', str(p / 'native-capture-plan.json'), '--worker-log-dir', str(target / 'worker_logs'),
            '--output', str(target / 'summary.json')], cwd=str(native_validator.parent), mode='cpu')]
    prechecks = [dict(key='native_lift32', kind='native', namespace=namespace, timeout_seconds=7200,
                     commands=commands, accepts_all_success_rates=True)]
    services = copy.deepcopy(old['services'])
    for service, port in zip(services, (18976, 18977)):
        argv = service['argv']
        assert argv[3] == str(service_donor)
        argv[3] = str(g / 'service/opendw_service_batched.py')
        position = argv.index('--t5-path')
        del argv[position:position + 2]
        for flag, value in [('--reward-checkpoint', contract['reward_checkpoint']), ('--port', str(port)),
                            ('--output-dir', str(O / 'services' / service['key'] / 'records'))]:
            assert argv.count(flag) == 1
            argv[argv.index(flag) + 1] = value
        service.update(url='http://127.0.0.1:' + str(port), cwd=str(g / 'service'),
            reward_checkpoint=contract['reward_checkpoint'], reward_checkpoint_sha256=contract['reward_checkpoint_sha256'])
        service['environment']['PYTHONPATH'] = str(g / 'service') + ':' + service['environment'].get('PYTHONPATH', '')
    trials = [dict(key=key, num_envs=64, episode_steps=length, config=str(p / (key + '.yaml')),
        config_sha256=sha(p / (key + '.yaml')), namespace='opendw_sz3_lift_' + key + '_v1', timeout_seconds=timeout)
        for key, length, timeout in [('startup_smoke', 32, 3600), ('formal', 384, 60 * 86400)]]
    scope = copy.deepcopy(old['graphics_scope'])
    scope.update(environment_fragment_file=str(p / 'graphics-fragment.json'),
                 environment_fragment_sha256=sha(p / 'graphics-fragment.json'))
    old_hook = owned(scope['post_borrow_hook_module'])
    assert sha(old_hook) == old['source_sha256'][str(old_hook)]
    native_hook = owned(c / 'native_prechecks.py')
    native_donor = owned(old['prechecks_module'])
    assert sha(native_donor) == old['source_sha256'][str(native_donor)]
    files = set([Path(__file__), patch_path, c / 'rm_adapter.py', native_hook, native_donor, old_hook, lifecycle,
                 cycle / 'plan.json', D / 'prepared-cycles/prepared-cycles.json', native_launcher, native_validator,
                 RM / 'code/rm_inference.py', Path(contract['reward_checkpoint']), Path(contract['reward_report']),
                 seeds_path, seeds_path.with_suffix('.receipt.json'), Path(formal['env']['train']['initial_state_path']),
                 Path(formal['env']['train']['initial_state_path']).with_suffix('.json')])
    files.update(p.glob('*.json'))
    files.update(p.glob('*.yaml'))
    files.update(g.rglob('*.py'))
    files.update(Path(row['target']) for row in copied)
    active = read(SCOPE / 'activation.json')
    scope_manifest = read(SCOPE / 'scope.json')
    files.update([SCOPE / 'activation.json', SCOPE / 'scope.json', Path(active['runtime_path']),
                  Path(active['environment_fragment_file']), Path(scope_manifest['bootstrap_path'])])
    for child in cp['children'].values():
        files.update([Path(child['path']) / 'plan.json', Path(child['module'])])
    runtime_files = ['rlinf/envs/__init__.py', 'rlinf/runners/embodied_runner.py',
                     'rlinf/envs/world_model/opendw_adapter.py', 'rlinf/envs/world_model/opendw_robotwin_env.py',
                     'examples/embodiment/train_embodied_agent.py']
    files.update(repo / name for name in runtime_files)
    files.update(NATIVE / name for name in ['rlinf/envs/__init__.py', 'rlinf/runners/embodied_runner.py',
        'rlinf/envs/robotwin/robotwin_env.py', 'rlinf/envs/robotwin/native_binary_recorder.py'])
    for path in files:
        if str(path) in old['source_sha256']:
            assert sha(path) == old['source_sha256'][str(path)], 'Frozen donor changed: ' + str(path)
    plan = dict(mode='multigpu_formal', start_mode='direct_start_user_override_20261004', startup_smoke=True,
        wm_batch_size=16, owner_dir=str(O), lifecycle_path=str(cycle), lifecycle_module=str(lifecycle),
        base_owner_module=str(g / 'owner/opendw_multigpu_owner.py'), python=cp['python'], repo=str(repo), repo_head=HEAD,
        physical_gpus=[4, 5, 6, 7], environment_file=str(p / 'environment.json'), services=services, trials=trials,
        prechecks=prechecks, prechecks_module=str(native_hook), native_prechecks_donor=str(native_donor),
        ray_address=cp['ray_address'], restore_wait_seconds=60, native_eval_seeds_sha256=sha(seeds_path),
        graphics_scope=scope, protocol_reference=dict(config=str(p / 'formal.yaml'), sha256=sha(p / 'formal.yaml')),
        budget=contract, source_sha256={str(path): sha(owned(path)) for path in sorted(files)},
        provenance=dict(source_owner=str(S / 'runs/click-bell-v2'), donor_plan_sha256=sha(DONOR / 'prepared/owner-plan.json'),
            original_sft=formal['actor']['model']['model_path'], policy_starts_from_original_sft=True,
            reward_checkpoint=contract['reward_checkpoint'], rm_generated_domain_verified=False,
            removed_prechecks=['native_rynn32', 'rynn32', 'native_bell32', 'bell_rm'],
            native_baseline='32 heldout seeds, C32/384, full native result required; no success-rate gate',
            startup_acceptance='Existing engineering exit/placement/cleanup checks; no one-block positive-reward or gradient gate',
            prepares_only=True))
    save(p / 'owner-plan.json', plan)
    wrapper = load('lift_formal_preflight', g / 'owner/opendw_formal_owner.py')
    module = wrapper.install(plan)
    module = load('lift_native_preflight', native_hook).install(module, plan)
    module.validate(plan)
    result = dict(prepared=True, cpu_validate_passed=True, processes_stopped=False, launched=False,
        owner_dir=str(O), plan=str(p / 'owner-plan.json'), plan_sha256=sha(p / 'owner-plan.json'),
        entrypoint=str(g / 'owner/opendw_formal_owner.py'), order=['native_lift32', 'startup_smoke', 'formal'],
        source_count=len(plan['source_sha256']), lifecycle=str(cycle))
    save(p / 'ready.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
