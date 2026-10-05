"""CPU prepare the bounded Rynn/native checks, then fresh click_bell WMRL.

No process is signalled or launched; the existing owner alone borrows/returns
the already-prepared RLT cycle when root explicitly starts it afterward.
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
import subprocess
import sys

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D = S / 'click-bell-v1'
C = D / 'code'
P = D / 'prepared'
G = D / 'generated'
R = S / 'rlinf-opendw-bell-v1'
NATIVE_REPO = S / 'rlinf-rynn-binary-v1'
OWNER = S / 'runs/click-bell-v1'
OLD_OWNER = S / 'runs/formal-b16-v1'
SCOPE = S / 'rynn-numeric-v1/rlt-return-repair-v2/scope'
CYCLE = S / 'rynn-binary-v1/prepared-cycles/cycle'
HEAD = '2151a08ee1bd75df1bef0d8190e594bd5c7f7977'
PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
RYNN_PY = '/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(D.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    path.chmod(0o600)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def command(script, *args, mode='cpu', environment_file=None, python=PY, cwd=None):
    result = dict(argv=[str(python), '-u', '-B', str(script), *map(str, args)],
                  cwd=str(cwd or Path(script).parent), mode=mode)
    if environment_file is not None:
        result['environment_file'] = str(environment_file)
    if mode == 'gpu':
        result['physical_gpu'] = 4
    return result


def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Prepare with CUDA hidden'
    assert not OWNER.exists(), 'Do not prepare over an already-created owner'
    assert not P.exists() and not G.exists() and not R.exists(), 'Fresh preparation paths required'
    old = read(OLD_OWNER / 'owner-plan.json')
    old_repo = Path(old['repo'])
    assert subprocess.check_output(['git', '-C', str(old_repo), 'rev-parse', 'HEAD'], text=True).strip() == HEAD == old['repo_head']
    assert subprocess.check_output(['git', '-C', str(NATIVE_REPO), 'rev-parse', 'HEAD'], text=True).strip() == HEAD
    assert 'native_binary_recorder' in (NATIVE_REPO / 'rlinf/envs/robotwin/robotwin_env.py').read_text()
    lifecycle = CYCLE / 'rlt_returned_multigpu_cycle.py'
    cp = read(CYCLE / 'plan.json')
    assert cp['physical_gpus'] == [4, 5, 6, 7] and not (CYCLE / 'rlt-stopped.json').exists()
    assert sha(lifecycle) == cp['script_sha256']
    asset = read(D / 'rm-asset.json')
    rm = Path(asset['checkpoint'])
    assert asset['strict_load'] is True and sha(rm) == asset['sha256']
    reset = D / 'assets/click_bell_clean50_reset.npz'
    reset_receipt = read(reset.with_suffix('.json'))
    assert reset_receipt['task'] == 'click_bell' and reset_receipt['count'] == 50 and sha(reset) == reset_receipt['sha256']
    base_path = S / 'formal-b16-control-v1/prepared/formal.yaml'
    base = read(base_path)
    assert base['env']['train']['task_name'] == 'adjust_bottle'
    original_sft = base['actor']['model']['model_path']
    assert Path(original_sft).is_dir()
    seed_source = old_repo / 'rlinf/envs/robotwin/seeds/eval_seeds.json'
    seed_record = copy.deepcopy(read(seed_source)['click_bell'])
    seed_values = seed_record['success_seeds']
    assert len(seed_values) >= 32 and len(set(seed_values)) == len(seed_values) and all(type(v) is int for v in seed_values)
    seed_record['success_seeds'] = seed_values[:32]
    required_code = [C / 'bell/patch_click_bell.py', C / 'bell/build_config.py', C / 'bell/probe_reward.py',
        C / 'ops/prechecks.py', C / 'ops/reuse_scope_hook.py', C / 'binary/prepare_native_eval.py',
        C / 'binary/run_native_eval.py', C / 'binary/merge_native_dataset.py', C / 'binary/rynn_binary_probe.py',
        C / 'binary/prepare_bell_reward_samples.py', C / 'binary/summarize_bell_reward_probe.py']
    assert all(path.is_file() for path in required_code)
    P.mkdir(mode=0o700)
    subprocess.run(['git', '-C', str(old_repo), 'worktree', 'add', '-b', 'codex/wmrl-click-bell-20261005', str(R), HEAD], check=True)
    dirty_paths = ['rlinf/envs/__init__.py', 'rlinf/runners/embodied_runner.py',
                   'rlinf/envs/world_model/opendw_adapter.py', 'rlinf/envs/world_model/opendw_robotwin_env.py']
    copied = {}
    for relative in dirty_paths:
        target = R / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_repo / relative, target)
        copied[relative] = sha(target)
    patch = load('bell_task_patcher', C / 'bell/patch_click_bell.py')
    G.mkdir(mode=0o700)
    generated = [
        (old_repo / dirty_paths[2], G / dirty_paths[2], patch.adapter_source),
        (old_repo / dirty_paths[3], G / dirty_paths[3], patch.env_source),
        (Path(old['base_owner_module']), G / 'owner/opendw_multigpu_owner.py', patch.base_owner_source),
        (Path(old['base_formal_wrapper']), G / 'owner/opendw_formal_owner.py', patch.formal_owner_source),
    ]
    generated_manifest = []
    for donor, target, transform in generated:
        assert sha(donor) == old['source_sha256'][str(donor)], 'Working donor changed: ' + str(donor)
        text = transform(donor.read_text())
        if target.name == 'opendw_formal_owner.py':
            anchor = "    if args.action == 'owner':\n        module.owner_main(plan)"
            assert text.count(anchor) == 1
            replacement = ("    if args.action == 'owner':\n"
                "        hook_path = Path(plan['prechecks_module'])\n"
                "        assert sha(hook_path) == plan['source_sha256'][str(hook_path)]\n"
                "        hook_spec = importlib.util.spec_from_file_location('bell_finite_prechecks', hook_path)\n"
                "        hook = importlib.util.module_from_spec(hook_spec)\n"
                "        hook_spec.loader.exec_module(hook)\n"
                "        module = hook.install(module, plan)\n"
                "        module.owner_main(plan)")
            text = text.replace(anchor, replacement, 1)
        compile(text, str(target), 'exec')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        generated_manifest.append(dict(donor=str(donor), donor_sha256=sha(donor), path=str(target), sha256=sha(target)))
    for relative in dirty_paths[2:]:
        shutil.copyfile(G / relative, R / relative)
    save(P / 'generated-sources.json', generated_manifest)
    seeds = P / 'click-bell-seeds.json'
    save(seeds, {'click_bell': seed_record})
    save(P / 'seed-receipt.json', dict(source=str(seed_source), source_sha256=sha(seed_source), task='click_bell',
        selection='first32 of existing task list; no new expert screening', count=32, sha256=sha(seeds)))
    builder = load('bell_config_builder', C / 'bell/build_config.py')
    formal, smoke, contract = builder.build(base, name='opendw-click-bell-b16-v1', owner_dir=str(OWNER),
        initial_state=str(reset), native_seeds=str(seeds), sft_path=original_sft,
        service_urls=['http://127.0.0.1:18956', 'http://127.0.0.1:18957'])
    save(P / 'formal.yaml', formal)
    save(P / 'startup_smoke.yaml', smoke)
    save(P / 'config-contract.json', contract)

    raw_env = read(old['environment_file'])
    base_env = {key: str(value).replace(str(old_repo), str(R)) for key, value in raw_env.items()}
    assert not any(key in base_env for key in ('CUDA_VISIBLE_DEVICES', 'LD_PRELOAD', 'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'))
    base_env.update(HOME='/home/chenyiteng', USER='chenyiteng', LOGNAME='chenyiteng', PYTHONDONTWRITEBYTECODE='1')
    save(P / 'environment.json', base_env)
    scoped = read(SCOPE / 'environment-fragment.json')
    scoped['PYTHONPATH'] = scoped['PYTHONPATH'].replace(str(old_repo), str(R))
    assert str(R) in scoped['PYTHONPATH'].split(':')
    save(P / 'graphics-fragment.json', scoped)
    native_env = {key: str(value).replace(str(R), str(NATIVE_REPO)) for key, value in dict(base_env, **scoped).items()}
    save(P / 'native-environment.json', native_env)
    native_builder = load('bell_native_config_builder', C / 'binary/prepare_native_eval.py')
    checkpoint = OLD_OWNER / 'formal/run/opendw-adjust-bottle-formal-b16-v1/checkpoints/global_step_70/actor/model_state_dict/full_weights.pt'
    assert checkpoint.is_file()
    for key, source, weights in [('native_rynn32', base, checkpoint), ('native_bell32', formal, None)]:
        native_cfg = native_builder.derive(source, OWNER / key, weights, 'opendw-' + key + '-1005-v1')
        save(P / (key + '.json'), native_cfg)

    binary = C / 'binary'
    prechecks = []
    for key, capture_mode, second in [
        ('native_rynn32', 'binary_terminal', command(binary / 'merge_native_dataset.py', '--capture-dir', OWNER / 'native_rynn32/capture', '--output-dir', OWNER / 'native_rynn32/dataset', '--expected-count', '32')),
        ('native_bell32', 'reward_native', command(binary / 'prepare_bell_reward_samples.py', '--capture-dir', OWNER / 'native_bell32/capture', '--output-dir', OWNER / 'native_bell32/dataset')),
    ]:
        namespace = 'opendw_sz3_' + key + '_1005_v1'
        first = command(binary / 'run_native_eval.py', '--config', P / (key + '.json'), '--receipt-dir', OWNER / key,
            '--private-repo', NATIVE_REPO, '--environment-fragment', P / 'native-environment.json',
            '--capture-dir', OWNER / key / 'capture', '--capture-mode', capture_mode, '--namespace', namespace,
            '--ray-address', cp['ray_address'], mode='ray', environment_file=P / 'native-environment.json', cwd=NATIVE_REPO)
        prechecks.append(dict(key=key, kind='native', namespace=namespace, timeout_seconds=7200, commands=[first, second]))
    rynn_cmd = command(binary / 'rynn_binary_probe.py', '--physical-gpu', '4', '--batch-size', '16',
        '--numeric-module', S / 'rynn-numeric-v1/code/rynn_numeric_probe.py',
        '--service-module', S / 'rynn-control-v2/code/rynn_success_service.py',
        '--official-inference', '/data/chenyiteng/projects/RynnValue-10e0d333/rynn_infer/inference.py',
        '--model-path', '/data/chenyiteng/models/RynnValue-8B-8738c5e4',
        '--manifest-path', '/data/chenyiteng/models/RynnValue-8B-8738c5e4/manifest.json',
        '--cases-json', OWNER / 'native_rynn32/dataset/cases.json', '--samples-npz', OWNER / 'native_rynn32/dataset/samples.npz',
        '--output', OWNER / 'rynn32/report.json', mode='gpu', python=RYNN_PY)
    prechecks.insert(1, dict(key='rynn32', kind='rynn_binary', timeout_seconds=1800, commands=[rynn_cmd], result_file=str(OWNER / 'rynn32/report.json')))
    # Resolve the actual service module directory, never guess a new model implementation.
    service_script = next(Path(arg) for arg in old['services'][0]['argv'] if Path(arg).name == 'opendw_service_batched.py')
    reward_module = service_script.parent
    assert (reward_module / 'opendw_reward.py').is_file()
    t5 = old['services'][0]['argv'][old['services'][0]['argv'].index('--t5-path') + 1]
    rm_python = old['services'][0]['argv'][0]
    bell_cmd = command(C / 'bell/probe_reward.py', '--samples', OWNER / 'native_bell32/dataset/samples.npz',
        '--checkpoint', rm, '--t5-path', t5, '--reward-module-dir', reward_module, '--physical-gpu', '4', '--batch-size', '16',
        '--output', OWNER / 'bell_rm/report.json', mode='gpu', python=rm_python)
    summary_cmd = command(binary / 'summarize_bell_reward_probe.py', '--samples-json', OWNER / 'native_bell32/dataset/samples.json',
        '--probe-result', OWNER / 'bell_rm/report.json', '--output', OWNER / 'bell_rm/summary.json')
    prechecks.append(dict(key='bell_rm', kind='bell_reward', timeout_seconds=1200, commands=[bell_cmd, summary_cmd], result_file=str(OWNER / 'bell_rm/report.json')))
    services = copy.deepcopy(old['services'])
    for service, port in zip(services, (18956, 18957)):
        argv = service['argv']
        for flag, value in [('--reward-checkpoint', str(rm)), ('--port', str(port)),
                            ('--output-dir', str(OWNER / 'services' / service['key'] / 'records'))]:
            assert argv.count(flag) == 1
            argv[argv.index(flag) + 1] = value
        service.update(url='http://127.0.0.1:' + str(port), reward_checkpoint=str(rm), reward_checkpoint_sha256=asset['sha256'])
        assert argv[argv.index('--wm-batch-size') + 1] == '16'
    trials = [dict(key=key, num_envs=64, episode_steps=length, config=str(P / (key + '.yaml')),
                   config_sha256=sha(P / (key + '.yaml')), namespace='opendw_sz3_bell_' + key + '_v1', timeout_seconds=timeout)
              for key, length, timeout in [('startup_smoke', 32, 3600), ('formal', 384, 60*86400)]]
    scope_manifest = read(SCOPE / 'scope.json')
    scope = dict(scope_dir=str(SCOPE), environment_fragment_file=str(P / 'graphics-fragment.json'),
        environment_fragment_sha256=sha(P / 'graphics-fragment.json'), post_borrow_hook_module=str(C / 'ops/reuse_scope_hook.py'),
        activation_receipt=str(SCOPE / 'activation.json'), profile_changes=False)
    files = set(required_code + [Path(__file__), lifecycle, CYCLE / 'plan.json', Path(scope_manifest['runtime_path']),
        Path(scope_manifest['bootstrap_path']), SCOPE / 'scope.json', SCOPE / 'activation.json',
        P / 'environment.json', P / 'graphics-fragment.json', P / 'native-environment.json',
        P / 'formal.yaml', P / 'startup_smoke.yaml', P / 'native_rynn32.json', P / 'native_bell32.json', seeds,
        D / 'rm-asset.json', reset.with_suffix('.json'),
        S / 'rynn-numeric-v1/code/rynn_numeric_probe.py', S / 'rynn-control-v2/code/rynn_success_service.py',
        Path('/data/chenyiteng/projects/RynnValue-10e0d333/rynn_infer/inference.py'),
        Path('/data/chenyiteng/models/RynnValue-8B-8738c5e4/manifest.json')])
    files.update(R / name for name in dirty_paths)
    files.update(G / name for name in ['owner/opendw_multigpu_owner.py', 'owner/opendw_formal_owner.py'])
    files.update(NATIVE_REPO / name for name in ['rlinf/envs/__init__.py', 'rlinf/runners/embodied_runner.py',
        'rlinf/envs/robotwin/robotwin_env.py', 'rlinf/envs/robotwin/native_binary_recorder.py'])
    for service in services:
        script = next(Path(arg) for arg in service['argv'] if Path(arg).name == 'opendw_service_batched.py')
        files.update(script.parent / name for name in ['opendw_service_batched.py', 'wm_batch.py', 'opendw_reward.py', 'opendw_action_telemetry.py'])
    for child in cp['children'].values():
        files.update([Path(child['path']) / 'plan.json', Path(child['module'])])
    plan = dict(mode='multigpu_formal', start_mode='direct_start_user_override_20261004', startup_smoke=True,
        wm_batch_size=16, owner_dir=str(OWNER), lifecycle_path=str(CYCLE), lifecycle_module=str(lifecycle),
        base_owner_module=str(G / 'owner/opendw_multigpu_owner.py'), python=cp['python'], repo=str(R), repo_head=HEAD,
        physical_gpus=[4, 5, 6, 7], environment_file=str(P / 'environment.json'), services=services, trials=trials,
        prechecks=prechecks, prechecks_module=str(C / 'ops/prechecks.py'), ray_address=cp['ray_address'],
        restore_wait_seconds=60, native_eval_seeds_sha256=sha(seeds), graphics_scope=scope,
        protocol_reference=dict(config=str(P / 'formal.yaml'), sha256=sha(P / 'formal.yaml')),
        source_sha256={str(path): sha(path) for path in sorted(files)}, budget=contract,
        provenance=dict(base_config=str(base_path), base_config_sha256=sha(base_path), copied_source_sha256=copied,
                        source_owner=str(OLD_OWNER), source_owner_plan_sha256=sha(OLD_OWNER / 'owner-plan.json'),
                        original_sft=original_sft, starts_from_policy_checkpoint=False, prepares_only=True))
    save(P / 'owner-plan.json', plan)
    save(P / 'ready.json', dict(time=datetime.datetime.now().astimezone().isoformat(), prepared=True, launched=False,
        plan=str(P / 'owner-plan.json'), sha256=sha(P / 'owner-plan.json'),
        entrypoint=str(G / 'owner/opendw_formal_owner.py'), repo=str(R), precheck_order=[r['key'] for r in prechecks]))
    print(json.dumps(read(P / 'ready.json')), flush=True)


if __name__ == '__main__':
    main()
