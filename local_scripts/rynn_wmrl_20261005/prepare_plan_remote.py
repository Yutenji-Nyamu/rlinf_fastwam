"""Prepare the reviewed Rynn plan only; never signal or launch any process."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D = S / 'rynn-control-v1'
P = D / 'prepared'
CODE = D / 'code'
OLD = S / 'runs/formal-b16-v1'
OWNER = S / 'runs/rynn-success-v1'
REPO = S / 'rlinf-rynn-v1'
RYNN_PY = '/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python'
MODEL = Path('/data/chenyiteng/models/RynnValue-8B-8738c5e4')
ROBOT = 'A dual-arm ALOHA robot with two grippers manipulating objects on a tabletop.'
CAMERA = 'A fixed head RGB camera observing both arms and the tabletop.'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-test-receipt', type=Path, default=P / 'cpu-tests.json')
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    tests = read(args.cpu_test_receipt)
    assert tests['all_cpu_tests_passed'] is True
    for path, digest in tests.get('source_sha256', {}).items():
        assert sha(path) == digest, 'Tested source changed: ' + path
    assert not (P / 'plan-template.json').exists() and not (P / 'ready.json').exists()
    prepared = read(P / 'base-prepared.json')
    assert prepared['repo'] == str(REPO) and prepared['runtime_untouched'] is True
    old = read(OLD / 'owner-plan.json')
    assert prepared['base_head'] == old['repo_head']
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == old['repo_head']
    resume_path = P / 'resume-receipt.json'
    resume = read(resume_path)
    assert resume['source_owner'] == str(OLD) and resume['completed_step'] == 70
    assert resume['complete'] is True
    old_row = next(r for r in old['trials'] if r['key'] == 'formal')
    assert sha(old_row['config']) == old_row['config_sha256']
    cfg = read(old_row['config'])
    original_name = cfg['runner']['logger']['experiment_name']
    new_name = 'opendw-adjust-bottle-rynn-success-v1'
    formal = json.loads(json.dumps(cfg).replace(str(OLD / 'formal'), str(OWNER / 'formal'))
                        .replace(original_name, new_name))
    formal['runner']['resume_dir'] = resume['checkpoint_path']
    formal['runner']['ckpt_path'] = None
    train = formal['env']['train']
    train.update(reward_source='rynn_success', rynn_invalid_reward_sentinel=-1.0,
        rynn_service_urls=['http://127.0.0.1:18954', 'http://127.0.0.1:18955'],
        rynn_run_id=OWNER.name + '/formal', rynn_batch_size_file=str(OWNER / 'rm_gate/result.json'),
        rynn_batch_size=8, rynn_timeout_s=7200, use_rel_reward=False)
    assert formal['runner']['max_steps'] == 200
    assert formal['actor']['global_batch_size'] == 2048 and formal['actor']['micro_batch_size'] == 8
    assert train['total_num_envs'] == 64 and train['rollout_epoch'] == train['group_size'] == 8
    assert train['chunk'] == 32 and train['max_episode_steps'] == train['max_steps_per_rollout_epoch'] == 384
    smoke = copy.deepcopy(formal)
    smoke['runner'].update(max_epochs=1, max_steps=1, save_interval=1, val_check_interval=-1,
                           resume_dir=None, ckpt_path=None)
    smoke['env']['train']['rollout_epoch'] = 1
    # 64 trajectories x 12 fixed chunks: one 768-sample global batch per U2 epoch.
    smoke['actor']['global_batch_size'] = 768
    smoke = json.loads(json.dumps(smoke).replace(str(OWNER / 'formal'), str(OWNER / 'startup_smoke'))
                       .replace(new_name, 'opendw-adjust-bottle-rynn-startup-v1'))
    smoke['env']['train']['rynn_run_id'] = OWNER.name + '/startup_smoke'
    trials = []
    for key, value, timeout in [('startup_smoke', smoke, 10800), ('formal', formal, 60 * 86400)]:
        path = P / (key + '.yaml')
        write(path, value)
        trials.append(dict(key=key, num_envs=64, episode_steps=384, config=str(path), config_sha256=sha(path),
            namespace='opendw_sz3_rynn_' + key + '_v1', timeout_seconds=timeout))
    services = copy.deepcopy(old['services'])
    for row in services:
        row['kind'] = 'wm'
        argv = row['argv']
        positions = [i for i, arg in enumerate(argv) if Path(arg).name == 'opendw_service_batched.py']
        assert len(positions) == 1
        argv[positions[0]] = str(CODE / 'opendw_service_batched.py')
        argv[argv.index('--output-dir') + 1] = str(OWNER / 'services' / row['key'] / 'records')
        row['cwd'] = str(CODE)
    for gpu, port in ((4, 18954), (5, 18955)):
        key = 'rm' + str(gpu)
        services.append(dict(key=key, kind='rynn_success', physical_gpu=gpu,
            url='http://127.0.0.1:' + str(port), cwd=str(CODE), startup_seconds=1200,
            environment=dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'),
            argv=[RYNN_PY, '-u', '-B', str(CODE / 'rynn_success_service.py'),
                '--model-path', str(MODEL), '--manifest-path', str(MODEL / 'manifest.json'),
                '--physical-gpu', str(gpu), '--port', str(port), '--batch-size', '16',
                '--robot-description', ROBOT, '--camera-description', CAMERA,
                '--log-path', str(OWNER / 'services' / key / 'records/infer.jsonl')]))
    environment = read(old['environment_file'])
    assert not any(key in environment for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
        'RLINF_OPENDW_GPU_SCOPE_MANIFEST', 'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST', 'LD_PRELOAD'))
    environment_path = P / 'environment.json'
    write(environment_path, environment)
    environment_path.chmod(0o600)
    source = dict(old['source_sha256'])
    source.update({str(path): sha(path) for path in CODE.glob('*.py')})
    changed_repo = ['rlinf/envs/__init__.py', 'rlinf/runners/embodied_runner.py',
        'rlinf/envs/world_model/opendw_adapter.py', 'rlinf/envs/world_model/opendw_robotwin_env.py',
        'rlinf/envs/world_model/rynn_success.py', 'rlinf/envs/world_model/rynn_actor_mask.py',
        'rlinf/workers/actor/embodied_fsdp_actor_worker.py']
    for relative in changed_repo:
        path = REPO / relative
        assert path.is_file(), 'New source was not deployed: ' + str(path)
        source[str(path)] = sha(path)
    for name in ('rynn_handoff.py', 'rynn_formal_owner.py', 'rynn_gate_client.py', 'rynn_success_service.py',
                 'opendw_service_batched.py', 'wm_batch.py', 'opendw_reward.py', 'opendw_action_telemetry.py'):
        assert str(CODE / name) in source, 'Required service support missing: ' + name
    scope = copy.deepcopy(old['graphics_scope'])
    active = read(scope['activation_receipt'])
    scope['reuse_active'] = dict(source_owner=str(OLD), source_owner_plan_sha256=sha(OLD / 'owner-plan.json'),
        activation_sha256=sha(scope['activation_receipt']), manifest_sha256=active['manifest_sha256'])
    source[active['runtime_path']] = sha(active['runtime_path'])
    samples = P / 'rm-sanity-clips.npz'
    assert samples.is_file()
    gate = dict(argv=[old['python'], '-u', '-B', str(CODE / 'rynn_gate_client.py'),
        '--urls', 'http://127.0.0.1:18954', 'http://127.0.0.1:18955', '--samples', str(samples),
        '--output', str(OWNER / 'rm_gate/result.json')], cwd=str(CODE), timeout_seconds=1800,
        output=str(OWNER / 'rm_gate/result.json'), samples_sha256=sha(samples))
    plan = dict(mode='multigpu_formal', start_mode='direct_start_user_override_20261004', startup_smoke=True,
        wm_batch_size=16, owner_dir=str(OWNER), lifecycle_path=old['lifecycle_path'],
        lifecycle_module=old['lifecycle_module'], base_owner_module=old['base_owner_module'],
        base_formal_wrapper=old['base_formal_wrapper'], python=old['python'], repo=str(REPO), repo_head=old['repo_head'],
        physical_gpus=[4, 5, 6, 7], environment_file=str(environment_path), source_sha256=source, services=services,
        trials=trials, restore_wait_seconds=60, native_eval_seeds_sha256=old['native_eval_seeds_sha256'],
        protocol_reference=old['protocol_reference'], graphics_scope=scope, rm_gate=gate,
        resume_checkpoint=dict(receipt=str(resume_path), sha256=sha(resume_path)),
        borrow_adoption=copy.deepcopy(old['borrow_adoption']), handoff_evidence=copy.deepcopy(old['handoff_evidence']),
        budget=dict(effective_runner_iterations=200, resumed_completed_iterations=70, remaining_iterations=130,
            N=64, G=8, R=8, C=32, L=384, wm_batch_size=16, actor_global_batch=2048, actor_microbatch=8,
            update_epoch=2, save_interval=10, native_eval_interval=10,
            reward='Rynn first Yes=1; No=0; unknown excludes whole G8 before advantage',
            startup_smoke=dict(N=64, G=8, R=1, L=384, global_batch=768, update_epoch=2,
                native_eval_interval=-1, initialization='original SFT; formal resumes verified CP70')))
    # Pure CPU contract check. This does not call owner_main, C.stop, or /onload.
    sys.path.insert(0, str(CODE))
    import rynn_formal_owner
    module = rynn_formal_owner.install(plan)
    module.validate(plan)
    ready = dict(schema=1, all_cpu_tests_passed=True, wm_batch_size=16,
        cpu_test_receipt=str(args.cpu_test_receipt), cpu_test_receipt_sha256=sha(args.cpu_test_receipt),
        source_sha256=source, created=datetime.datetime.now().astimezone().isoformat())
    write(P / 'plan-template.json', plan)
    write(P / 'ready.json', ready)
    print(json.dumps(dict(plan=str(P / 'plan-template.json'), ready=str(P / 'ready.json'),
        old_runtime_untouched=True, resumed_completed_iterations=70, formal_remaining_iterations=130)))


if __name__ == '__main__':
    main()
