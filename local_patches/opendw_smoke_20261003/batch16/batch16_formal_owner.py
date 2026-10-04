"""Batch-16 service wrapper over the frozen formal-v2 owner.

The startup smoke starts from SFT. Formal alone resumes the separately verified
checkpoint; neither its optimizer nor its runner step comes from short smoke.
The already active four-card graphics scope is verified and reused unchanged.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path('/data/chenyiteng')
SOURCE_OWNER = ROOT / 'projects/opendw-robotwin-smoke-20261003/runs/formal-v2'
THIS = Path(__file__).absolute()


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


def owned(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
    assert not path.is_symlink() and path.stat().st_uid == 20001
    return path


def verify_resume(receipt_path, expected_sha256=None):
    path = owned(receipt_path)
    if expected_sha256 is not None:
        assert sha(path) == expected_sha256, 'Resume receipt changed'
    value = read(path)
    assert value['schema'] == 1 and value['kind'] == 'opendw-formal-resume-checkpoint'
    assert Path(value['source_owner']).resolve() == SOURCE_OWNER.resolve()
    assert sha(SOURCE_OWNER / 'owner-plan.json') == value['source_owner_plan_sha256']
    assert value['complete'] is True and value['actor_world_size'] == 2
    step = value['completed_step']
    assert type(step) is int and 0 < step < 200
    checkpoint = owned(value['checkpoint_path'])
    assert checkpoint.is_dir() and checkpoint.name == 'global_step_' + str(step)
    assert checkpoint.resolve().is_relative_to(SOURCE_OWNER.resolve())
    runner = value['runner_state']
    assert runner['global_step'] == step
    assert runner['optimizer_state_saved'] is True and runner['scheduler_state_saved'] is True
    assert [row['rank'] for row in value['ranks']] == [0, 1]
    seen = set()
    for rank in value['ranks']:
        assert rank['complete'] is True and rank['files']
        for file in rank['files']:
            target = owned(file['path'])
            assert target.is_file() and target.resolve().is_relative_to(checkpoint.resolve())
            assert str(target.resolve()) not in seen, 'Checkpoint rank files overlap'
            seen.add(str(target.resolve()))
            stat = target.stat()
            assert stat.st_size == file['bytes'] > 0 and stat.st_mtime_ns == file['mtime_ns']
            if file.get('sha256'):
                assert sha(target) == file['sha256']
    full = value['full_weights']
    target = owned(full['path'])
    assert target.resolve() == (checkpoint / 'actor/model_state_dict/full_weights.pt').resolve()
    stat = target.stat()
    assert stat.st_size == full['bytes'] > 0 and stat.st_mtime_ns == full['mtime_ns']
    if full.get('sha256'):
        assert sha(target) == full['sha256']
    return value


def verify_scope_reuse(plan):
    scope = plan['graphics_scope']
    reuse = scope['reuse_active']
    assert Path(reuse['source_owner']).resolve() == SOURCE_OWNER.resolve()
    assert sha(SOURCE_OWNER / 'owner-plan.json') == reuse['source_owner_plan_sha256']
    old = read(SOURCE_OWNER / 'owner-plan.json')['graphics_scope']
    for key in ('scope_dir', 'environment_fragment_file', 'environment_fragment_sha256',
                'post_borrow_hook_module', 'prepare_module', 'activation_receipt'):
        assert scope[key] == old[key], 'Scope reuse must keep its exact original inputs: ' + key
    active_path = owned(scope['activation_receipt'])
    assert sha(active_path) == reuse['activation_sha256']
    active = read(active_path)
    assert active['status'] == 'active' and active['uid'] == 20001
    assert active['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    fragment_path = owned(scope['environment_fragment_file'])
    assert sha(fragment_path) == scope['environment_fragment_sha256']
    fragment = read(fragment_path)
    assert active['environment_fragment'] == fragment
    manifest_path = owned(fragment['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])
    assert sha(manifest_path) == active['manifest_sha256'] == reuse['manifest_sha256']
    runtime = owned(active['runtime_path'])
    assert sha(runtime) == active['runtime_sha256']
    assert plan['source_sha256'][str(runtime)] == sha(runtime)
    module = load('batch16_reused_graphics_runtime', runtime)
    manifest = module.read_manifest(manifest_path)
    assert manifest['token'] == active['scope_id']
    assert set(manifest['cards']) == {'4', '5', '6', '7'}
    return active, fragment, manifest_path


def smoke_batch_gate(plan):
    """Require actual successful B16 denoiser calls on both WM GPUs."""
    owner = Path(plan['owner_dir'])
    assert read(owner / 'startup_smoke/driver-finished.json')['exit_code'] == 0
    assert (owner / 'startup_smoke/verified-placement.json').is_file()
    evidence = []
    for service in plan['services']:
        argv = service['argv']
        directory = Path(argv[argv.index('--output-dir') + 1])
        path = owned(directory / 'service-events.jsonl')
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        matched = []
        for index, row in enumerate(rows):
            if row.get('event') != 'batch_completed' or row.get('actual_wm_batch') != 16:
                continue
            assert row['configured_wm_batch'] == 16 and row['execution_mode'] == 'batched'
            assert row['outputs_finite'] is True and row['rows_completed'] == 16
            proof = row['kernel_proof']
            assert proof['kernel'] == 'infer_joint_batch' and proof['batch_size'] == 16
            assert proof['denoiser_batch_sizes'] == [16] * 10, 'B16 was not sustained through all ten denoiser steps'
            assert proof['decoded_video_shape'][0] == proof['action_shape'][0] == 16
            matched.append(dict(line=index + 1, actual_wm_batch=16, kernel_proof=proof))
        assert matched, 'No actual successful B16 kernel call in startup smoke: ' + service['key']
        evidence.append(dict(service=service['key'], physical_gpu=service['physical_gpu'], path=str(path),
            prefix_bytes=len(data), prefix_sha256=hashlib.sha256(data).hexdigest(), verified_batches=matched))
    return dict(schema=1, status='passed', wm_batch_size=16, services=evidence,
        formal_resume_receipt=plan['resume_checkpoint'], startup_checkpoint_used_for_formal=False)


def verify_saved_gate(plan):
    gate = read(owned(Path(plan['owner_dir']) / 'batch16-smoke-gate.json'))
    assert gate['status'] == 'passed' and gate['wm_batch_size'] == 16
    assert gate['formal_resume_receipt'] == plan['resume_checkpoint']
    assert gate['startup_checkpoint_used_for_formal'] is False
    assert [row['physical_gpu'] for row in gate['services']] == [6, 7]
    for evidence in gate['services']:
        with owned(evidence['path']).open('rb') as stream:
            data = stream.read(evidence['prefix_bytes'])
        assert len(data) == evidence['prefix_bytes'] and hashlib.sha256(data).hexdigest() == evidence['prefix_sha256']
    return gate


def install(plan):
    base = owned(plan['base_formal_wrapper'])
    assert sha(base) == plan['source_sha256'][str(base)]
    old = read(SOURCE_OWNER / 'owner-plan.json')
    assert sha(base) == old['owner_script_sha256']
    W = load('batch16_frozen_formal_wrapper', base)
    W.THIS = THIS
    original_formal = W.validate_formal_config
    original_smoke = W.validate_startup_smoke
    original_validate = W.validate

    def formal_with_resume(cfg, row, current, owner, reference):
        evidence = current['resume_checkpoint']
        resumed = verify_resume(evidence['receipt'], evidence['sha256'])
        assert cfg['runner']['resume_dir'] == resumed['checkpoint_path']
        assert cfg['runner'].get('ckpt_path') is None
        shadow = copy.deepcopy(cfg)
        shadow['runner']['resume_dir'] = None
        original_formal(shadow, row, current, owner, reference)

    def smoke_from_sft(cfg, row, current, owner, formal):
        assert cfg['runner'].get('resume_dir') is None and cfg['runner'].get('ckpt_path') is None
        shadow = copy.deepcopy(formal)
        shadow['runner']['resume_dir'] = None
        original_smoke(cfg, row, current, owner, shadow)

    def validate(current, frozen=False):
        assert current['startup_smoke'] is True and current['wm_batch_size'] == 16
        for service in current['services']:
            argv = service['argv']
            assert argv.count('--wm-batch-size') == 1
            assert argv[argv.index('--wm-batch-size') + 1] == '16'
            scripts = [Path(arg) for arg in argv if Path(arg).name == 'opendw_service_batched.py']
            assert len(scripts) == 1
            assert current['source_sha256'][str(scripts[0])] == sha(owned(scripts[0]))
        verify_scope_reuse(current)
        return original_validate(current, frozen)

    def reuse_after_borrow(current):
        M = W.M
        assert W.verify_adoption(current)
        assert not M.H.same(read(SOURCE_OWNER / 'owner-identity.json'))
        assert not M.H.gpu_processes([4, 5, 6, 7]), 'Previous WM contexts remain'
        active, fragment, manifest = verify_scope_reuse(current)
        proof = dict(time=M.H.now(), status='scope_activated', native_probe_verified=False,
            native_env_reset_completed=False, native_success_evaluated=False,
            base_environment_sha256=sha(current['environment_file']),
            environment_fragment_sha256=current['graphics_scope']['environment_fragment_sha256'],
            scope_manifest_path=str(manifest), scope_manifest_sha256=sha(manifest),
            activation_receipt=current['graphics_scope']['activation_receipt'],
            activation_receipt_sha256=sha(current['graphics_scope']['activation_receipt']),
            reused_active_scope=True, source_owner=str(SOURCE_OWNER))
        W.validate_native_receipt(current, proof)
        M.record(Path(current['owner_dir']) / 'scope-activated.json', proof)
        assert W.OWNER_ENV is not None
        W.OWNER_ENV.update(fragment)

    W.validate_formal_config = formal_with_resume
    W.validate_startup_smoke = smoke_from_sft
    W.validate = validate
    W.post_borrow = reuse_after_borrow
    M = W.install(plan)
    original_record, original_driver = M.record, M.run_driver

    def record_with_batch_gate(path, value):
        original_record(path, value)
        path = Path(path)
        if path == Path(plan['owner_dir']) / 'startup_smoke/result.json' and value['exit_code'] == 0:
            gate = smoke_batch_gate(plan)
            gate['time'] = M.H.now()
            original_record(Path(plan['owner_dir']) / 'batch16-smoke-gate.json', gate)

    def driver_after_batch_gate(frozen_plan, key):
        if key == 'formal':
            verify_saved_gate(frozen_plan)
        return original_driver(frozen_plan, key)

    M.record = record_with_batch_gate
    M.run_driver = driver_after_batch_gate
    return M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('action', choices=('owner', 'driver'))
    parser.add_argument('--key')
    args = parser.parse_args()
    plan = read(args.plan)
    module = install(plan)
    if args.action == 'owner':
        module.owner_main(plan)
    else:
        assert args.key in ('startup_smoke', 'formal')
        module.run_driver(plan, args.key)


if __name__ == '__main__':
    main()
