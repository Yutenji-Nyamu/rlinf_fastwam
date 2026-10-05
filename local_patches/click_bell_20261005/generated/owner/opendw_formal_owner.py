"""Thin formal entry point over the frozen four-GPU smoke lifecycle/driver.

Only this independent wrapper changes scheduling validation and formal evidence
gates. The imported owner retains its exact PID/namespace cleanup, CPU-first WM
startup, four-card placement assertions, and RLT return behavior.
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


M = None
DONOR = None
THIS = Path(__file__).absolute()
OWNER_ENV = None
SCOPE_KEYS = {'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST', 'PYTHONPATH', 'LD_PRELOAD',
              '__GL_APPLICATION_PROFILE', '__GL_APPLICATION_PROFILE_LOG', 'HOME', 'USER', 'LOGNAME'}
DIRECT_START = 'direct_start_user_override_20261004'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(cfg):
    value = copy.deepcopy({k: cfg[k] for k in ('actor', 'algorithm', 'rollout', 'weight_syncer', 'reward', 'critic', 'cluster')})
    value['train'] = copy.deepcopy(cfg['env']['train'])
    value['actor'].pop('group_name', None)
    value['rollout'].pop('group_name', None)
    value['train'].pop('service_urls', None)
    value['train']['video_cfg'].pop('video_base_dir', None)
    value['algorithm']['dvac_gradient_weighting'].pop('output_dir', None)
    return value


def scope_fragment(plan):
    path = M.owned_path(plan['graphics_scope']['environment_fragment_file'])
    assert sha(path) == plan['graphics_scope']['environment_fragment_sha256']
    value = read(path)
    assert set(value) == SCOPE_KEYS and all(isinstance(v, str) and v for v in value.values())
    assert value['__GL_APPLICATION_PROFILE'] == '1'
    assert Path(value['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST']).is_absolute()
    assert not any(k in value for k in M.MASKS)
    return value


def validate_native_receipt(plan, proof):
    if plan.get('start_mode') == DIRECT_START:
        assert proof['status'] == 'scope_activated'
        assert proof['native_probe_verified'] is False
        assert proof['base_environment_sha256'] == sha(plan['environment_file'])
        assert proof['environment_fragment_sha256'] == plan['graphics_scope']['environment_fragment_sha256']
        manifest = M.owned_path(scope_fragment(plan)['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])
        assert proof['scope_manifest_path'] == str(manifest) and proof['scope_manifest_sha256'] == sha(manifest)
        return
    assert proof['native_env_reset_completed'] is True and proof['physical_gpus'] == [6, 7]
    assert proof['owned_outside_gpu4_7_contexts'] == []
    assert proof['post_native_probe_all_workers_stopped'] is True
    assert proof['base_environment_sha256'] == sha(plan['environment_file'])
    assert proof['environment_fragment_sha256'] == plan['graphics_scope']['environment_fragment_sha256']
    assert proof['native_eval_seeds_sha256'] == plan['native_eval_seeds_sha256']
    manifest = M.owned_path(scope_fragment(plan)['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])
    assert proof['scope_manifest_path'] == str(manifest) and proof['scope_manifest_sha256'] == sha(manifest)


def post_borrow(plan):
    """Called after exact RLT stop, before any new formal/native worker launch."""
    assert (Path(plan['lifecycle_path']) / 'rlt-stopped.json').is_file()
    assert not M.H.gpu_processes([4, 5, 6, 7]), 'Borrowed cards must be empty before scope installation'
    module_path = M.owned_path(plan['graphics_scope']['post_borrow_hook_module'])
    assert sha(module_path) == plan['source_sha256'][str(module_path)]
    spec = importlib.util.spec_from_file_location('formal_post_borrow_scope', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Direct start activates only the audited graphics profile. The formal
    # driver's real native evaluation initialization supplies the first test.
    proof = module.install_and_verify(plan)
    validate_native_receipt(plan, proof)
    assert not M.H.gpu_processes([4, 5, 6, 7]), 'Scope installation must not create GPU contexts'
    receipt_name = 'scope-activated.json' if plan.get('start_mode') == DIRECT_START else 'native-evaluation-verified.json'
    M.record(Path(plan['owner_dir']) / receipt_name, proof)
    assert OWNER_ENV is not None
    # WM services were already started CPU-first using the unscoped base env.
    # Only subsequently launched formal drivers inherit this in-memory merge.
    OWNER_ENV.update(scope_fragment(plan))


def learning_gate(plan):
    gate = plan['learning_evidence']
    report_path = M.owned_path(gate['path'])
    assert sha(report_path) == gate['sha256'], 'Learning audit changed'
    report = read(report_path)
    assert report['analysis_kind'] == 'opendw_multigpu_smoke_analysis_v1'
    rows = [r for r in report['trials'] if r['key'] == gate['trial_key']]
    assert len(rows) == 1
    row = rows[0]
    assert row['learning_status'] == 'effective_update_verified', 'No verified reward-driven optimizer update'
    assert row['binding_verified'] is True and row['exit_code'] == 0
    rec, tb, cp = row['reconstruction'], row['tensorboard'], row['checkpoint']
    assert rec['status'] == 'reconstructed' and rec['trajectories'] == 512 and rec['maximum_chunks'] == 6144
    assert rec['unknown_groups'] == 0 and rec['retained_groups'] > 0 and rec['nonzero_advantage_groups'] > 0
    assert rec['valid_chunks_after_filter'] > 0
    assert tb['finite'] and tb['stable'] and tb['nonzero_grad'] and tb['nonzero_advantage']
    assert cp['status'] == 'checked' and cp['finite_nonzero_update_state'] and cp['cuda_initialized'] is False
    assert [r['rank'] for r in cp['ranks']] == [0, 1]
    for rank in cp['ranks']:
        assert rank['stable'] and rank['model']['finite'] and rank['optimizer']['finite']
        assert rank['positive_lr_nonzero_adam_moments']
        assert rank['adam_step_min'] == rank['adam_step_max'] == 6, 'Long smoke must perform six scheduled Adam steps'
        assert M.owned_path(rank['path']).stat().st_size == rank['bytes']
    assert row['resources']['owned_outside_gpu4_7_contexts'] == []
    final_path = M.owned_path(gate['owner_final_path'])
    assert sha(final_path) == gate['owner_final_sha256']
    final = read(final_path)
    assert final['mode'] == 'multigpu_smoke' and final['terminal_status'] == 'completed' and final['error'] is None
    # RLT first-round acceptance is intentionally not a WM launch gate. A
    # recorded return issue needs a separately proven new lifecycle, not edits
    # to the old final receipt; C.load_plan validates that new ownership chain.
    assert next(r for r in final['trials'] if r['key'] == gate['trial_key'])['exit_code'] == 0
    smoke_config = M.owned_path(gate['smoke_config'])
    assert sha(smoke_config) == gate['smoke_config_sha256']
    assert report['source_sha256'][str(smoke_config)] == gate['smoke_config_sha256']
    return M.H.config(smoke_config)


def protocol_reference(plan):
    if plan.get('start_mode') != DIRECT_START:
        return learning_gate(plan)
    reference = plan['protocol_reference']
    path = M.owned_path(reference['config'])
    assert sha(path) == reference['sha256'], 'Direct-start source configuration changed'
    # This authorization changes the acceptance gate, never the algorithm.
    # No CP, gradient, advantage, or valid-group requirement is implied here.
    return M.H.config(path)


def verify_adoption(plan):
    if not plan.get('borrow_adoption'):
        assert not (Path(plan['lifecycle_path']) / 'rlt-stopped.json').exists(), 'Unproven existing borrow'
        return False
    assert plan.get('start_mode') == DIRECT_START
    adoption = plan['borrow_adoption']
    path = M.owned_path(adoption['receipt'])
    assert sha(path) == adoption['sha256'], 'Borrow-adoption receipt changed'
    assert (Path(plan['lifecycle_path']) / 'rlt-stopped.json').is_file()
    # The lifecycle owns the stopped-driver/CP/namespace provenance. Its new
    # explicit validator is mandatory; do not infer adoption from file presence.
    M.C.verify_adoption(Path(plan['lifecycle_path']), path)
    return True


def validate_formal_config(cfg, row, plan, owner, smoke_cfg):
    assert row['key'] == 'formal' and row['episode_steps'] == 384 and row['num_envs'] == 64
    assert protocol(cfg) == protocol(smoke_cfg), 'Formal training protocol differs from the frozen full configuration'
    runner = cfg['runner']
    assert (runner['max_epochs'], runner['max_steps']) == (1000, 200), 'Restore actual Control cap: 200 runner iterations'
    assert runner['save_interval'] == runner['val_check_interval'] == 10
    assert runner.get('resume_dir') is None and runner.get('ckpt_path') is None
    assert 0 < row['timeout_seconds'] <= 60 * 86400, 'Explicit overall clock bound is required'
    # Reuse the exact smoke training-contract checks on a schedule-only shadow;
    # no shadow is written, launched or passed to training.
    shadow, shadow_row = copy.deepcopy(cfg), copy.deepcopy(row)
    shadow['runner'].update(max_epochs=1, max_steps=1, save_interval=1, val_check_interval=-1)
    shadow_row['timeout_seconds'] = min(row['timeout_seconds'], 10800)
    M.validate_config(shadow, shadow_row, plan, owner)
    native = cfg['env']['eval']
    assert native['env_type'] == 'robotwin' and native['is_eval'] is True
    assert native['task_config']['task_name'] == 'click_bell'
    assert native['total_num_envs'] == 32 and native['rollout_epoch'] == native['group_size'] == 1
    assert native['max_episode_steps'] == native['max_steps_per_rollout_epoch'] == native['task_config']['step_lim'] == 384
    assert native['enable_offload'] is True and native['use_fixed_reset_state_ids'] is True
    assert native['auto_reset'] is True and native['ignore_terminations'] is True
    assert native['task_config']['embodiment'] == ['aloha-agilex']
    assert native['task_config']['camera']['collect_head_camera'] is True
    assert native['task_config']['camera']['collect_wrist_camera'] is True
    assert native['center_crop'] is False
    assert Path(native['assets_path']).is_dir()
    assert sha(M.owned_path(native['seeds_path'])) == plan['native_eval_seeds_sha256']
    values = read(native['seeds_path'])['click_bell']['success_seeds']
    assert len(values) >= 32 and len(set(values)) == len(values) and all(type(v) is int for v in values)


def startup_contract(cfg):
    """Ignore only run identities and output paths when comparing both trials."""
    value = copy.deepcopy(cfg)
    for path in (
        ('env', 'group_name'), ('actor', 'group_name'), ('rollout', 'group_name'),
        ('runner', 'logger', 'log_path'), ('runner', 'logger', 'experiment_name'),
        ('runner', 'per_worker_log_path'),
        ('env', 'train', 'video_cfg', 'video_base_dir'),
        ('env', 'eval', 'video_cfg', 'video_base_dir'),
        ('env', 'eval', 'task_config', 'save_path'),
        ('algorithm', 'dvac_gradient_weighting', 'output_dir'),
    ):
        parent = value
        for key in path[:-1]:
            parent = parent[key]
        parent.pop(path[-1], None)
    return value


def validate_startup_smoke(cfg, row, plan, owner, formal_cfg):
    """Same parallel placement/model as formal; one short training/eval step."""
    assert row['key'] == 'startup_smoke' and row['episode_steps'] == 32 and row['num_envs'] == 64
    runner, train, native = cfg['runner'], cfg['env']['train'], cfg['env']['eval']
    assert runner['max_epochs'] == runner['max_steps'] == 1
    assert runner['save_interval'] == runner['val_check_interval'] == 1
    assert train['rollout_epoch'] == 1
    assert train['max_episode_steps'] == train['max_steps_per_rollout_epoch'] == 32
    assert native['max_episode_steps'] == native['max_steps_per_rollout_epoch'] == native['task_config']['step_lim'] == 32
    assert cfg['actor']['global_batch_size'] == 64
    assert 0 < row['timeout_seconds'] <= 10800
    # Reconstruct the formal contract by restoring exactly the authorized
    # serial-work changes. Everything else, including native N32/three views,
    # action transforms, services, actor microbatch and two-rank placement,
    # must match. Neither config passed to training is mutated here.
    restored = copy.deepcopy(cfg)
    for name in ('max_epochs', 'max_steps', 'save_interval', 'val_check_interval'):
        restored['runner'][name] = formal_cfg['runner'][name]
    for name in ('rollout_epoch', 'max_episode_steps', 'max_steps_per_rollout_epoch'):
        restored['env']['train'][name] = formal_cfg['env']['train'][name]
    for name in ('max_episode_steps', 'max_steps_per_rollout_epoch'):
        restored['env']['eval'][name] = formal_cfg['env']['eval'][name]
    restored['env']['eval']['task_config']['step_lim'] = formal_cfg['env']['eval']['task_config']['step_lim']
    restored['actor']['global_batch_size'] = formal_cfg['actor']['global_batch_size']
    assert startup_contract(restored) == startup_contract(formal_cfg), 'Startup smoke differs beyond its serial-work/output allowance'
    # The donor checks its older R8/GB512 short-trial schema. Validate an
    # in-memory shadow to inherit those checks without changing actual R1/64.
    shadow = copy.deepcopy(cfg)
    shadow['runner']['val_check_interval'] = -1
    shadow['env']['train']['rollout_epoch'] = 8
    shadow['actor']['global_batch_size'] = 512
    M.validate_config(shadow, row, plan, owner)


def trial_keys(plan):
    assert type(plan.get('startup_smoke', False)) is bool
    expected = ['startup_smoke', 'formal'] if plan.get('startup_smoke', False) else ['formal']
    assert [row['key'] for row in plan['trials']] == expected, 'Require the declared startup smoke before formal'
    assert len({row['namespace'] for row in plan['trials']}) == len(expected)
    assert len({row['config'] for row in plan['trials']}) == len(expected)
    return expected


def validate(plan, frozen=False):
    global OWNER_ENV
    assert os.getuid() == M.UID and socket.gethostname() == 'h100-gpu01'
    M.pidfd_probe()
    owner = M.owned_path(plan['owner_dir'], exists=frozen)
    cycle = M.owned_path(plan['lifecycle_path'])
    assert plan['python'] == M.C.load_plan(cycle)['python']
    repo = M.owned_path(plan['repo'])
    assert 'opendw' in str(repo).lower()
    assert M.subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == plan['repo_head']
    assert plan['mode'] == 'multigpu_formal' and plan['physical_gpus'] == [4, 5, 6, 7]
    assert plan['source_sha256'] and str(DONOR) in plan['source_sha256'] and str(THIS) in plan['source_sha256']
    for path, digest in plan['source_sha256'].items():
        assert sha(M.owned_path(path)) == digest, 'Reviewed source changed: ' + path
    trial_keys(plan)
    smoke_cfg = protocol_reference(plan)
    formal_row = plan['trials'][-1]
    formal_cfg = M.H.config(M.owned_path(formal_row['config']))
    validate_formal_config(formal_cfg, formal_row, plan, owner, smoke_cfg)
    for row in plan['trials']:
        assert row['namespace'].startswith('opendw_') and re.fullmatch(r'[A-Za-z0-9_-]+', row['namespace'])
        cfg = formal_cfg if row['key'] == 'formal' else M.H.config(M.owned_path(row['config']))
        if row['key'] == 'startup_smoke':
            validate_startup_smoke(cfg, row, plan, owner, formal_cfg)
        M.owned_path(cfg['runner']['logger']['log_path'], False)
    M.validate_services(plan['services'], owner)
    assert 0 < plan.get('restore_wait_seconds', 60) <= 60
    environment = read(M.owned_path(plan['environment_file']))
    assert not any(k in environment for k in M.MASKS)
    fragment = scope_fragment(plan)  # Immutable CPU plan; installed manifest may not exist yet.
    hook = M.owned_path(plan['graphics_scope']['post_borrow_hook_module'])
    assert sha(hook) == plan['source_sha256'][str(hook)]
    if frozen:
        receipt_name = 'scope-activated.json' if plan.get('start_mode') == DIRECT_START else 'native-evaluation-verified.json'
        validate_native_receipt(plan, read(M.owned_path(Path(plan['owner_dir']) / receipt_name)))
        environment.update(fragment)
        for row in plan['trials']:
            assert row['config_sha256'] == sha(row['config'])
        assert plan['owner_script_sha256'] == sha(THIS)
        assert plan['environment_sha256'] == sha(plan['environment_file'])
        assert plan['lifecycle_plan_sha256'] == sha(cycle / 'plan.json')
    else:
        assert not any(k in environment for k in SCOPE_KEYS - {'PYTHONPATH', 'HOME', 'USER', 'LOGNAME'}), \
            'Base services must not enable a new/old scope before the post-borrow hook'
        OWNER_ENV = environment
    return owner, cycle, repo, environment


def install(plan):
    global M, DONOR, THIS
    # The donor launches its child through Path(__file__).resolve(). Recover
    # the one pinned spelling after /data -> backing-mount canonicalization.
    entrypoints = [Path(path) for path, digest in plan['source_sha256'].items()
                   if Path(path).resolve() == THIS.resolve() and digest == sha(THIS)]
    assert len(entrypoints) == 1, 'Formal entrypoint must have one frozen source identity'
    THIS = entrypoints[0]
    DONOR = Path(plan['base_owner_module'])
    assert DONOR.is_absolute() and DONOR.is_file()
    assert sha(DONOR) == plan['source_sha256'][str(DONOR)], 'Base owner is not the pinned source'
    spec = importlib.util.spec_from_file_location('frozen_multigpu_owner', DONOR)
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    # The reused owner freezes/executes its entrypoint via __file__. Point that
    # entrypoint to this wrapper so child drivers receive these same validators.
    M.__file__ = str(THIS)
    M.validate = validate
    M.load_lifecycle(plan)
    original_stop = M.C.stop
    def stopped_then_native_scope(stage):
        if plan.get('borrow_adoption'):
            assert verify_adoption(plan)
            result = read(Path(stage) / 'rlt-stopped.json')
        else:
            result = original_stop(stage)
        frozen_plan = read(Path(plan['owner_dir']) / 'owner-plan.json')
        post_borrow(frozen_plan)
        return result
    M.C.stop = stopped_then_native_scope
    if plan.get('borrow_adoption'):
        # Change only the two exact lifecycle entry conditions. All cleanup,
        # namespace, PID, failure-return, monitoring and launch logic remains
        # the frozen donor's implementation, with an auditable source delta.
        source = inspect.getsource(M.owner_main)
        old_assert = "    assert not (cycle/'rlt-stopped.json').exists(), 'Owner must witness its own borrowing'"
        old_borrowed = '\n    borrowed = False\n'
        assert source.count(old_assert) == 1 and source.count(old_borrowed) == 1
        replacement = source.replace(old_assert, '    adopted_borrow = verify_formal_adoption(input_plan)', 1)
        replacement = replacement.replace(old_borrowed, '\n    borrowed = adopted_borrow\n', 1)
        M.verify_formal_adoption = verify_adoption
        M.FORMAL_ADOPTION_SOURCE_DELTA = dict(donor_sha256=sha(DONOR),
            function_before_sha256=hashlib.sha256(source.encode()).hexdigest(),
            function_after_sha256=hashlib.sha256(replacement.encode()).hexdigest(),
            replacements=[dict(before=old_assert, after='    adopted_borrow = verify_formal_adoption(input_plan)'),
                          dict(before=old_borrowed, after='\n    borrowed = adopted_borrow\n')])
        exec(compile(replacement, str(THIS) + ':audited-adopted-owner-main', 'exec'), M.__dict__)
        original_record = M.record
        def record_with_delta(path, value):
            original_record(path, value)
            if Path(path).name == 'owner-plan.json':
                original_record(Path(path).parent / 'formal-owner-source-delta.json', M.FORMAL_ADOPTION_SOURCE_DELTA)
        M.record = record_with_delta
    original_driver = M.run_driver
    def driver_with_ray_scope(frozen_plan, key):
        import ray
        ray_init = ray.init
        def inject_scope(*args, **kwargs):
            environment = kwargs.setdefault('runtime_env', {}).setdefault('env_vars', {})
            for name in SCOPE_KEYS:
                assert os.environ.get(name), 'Formal child did not inherit scope variable: ' + name
                environment[name] = os.environ[name]
            return ray_init(*args, **kwargs)
        ray.init = inject_scope
        try:
            return original_driver(frozen_plan, key)
        finally:
            ray.init = ray_init
    M.run_driver = driver_with_ray_scope
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
        hook_path = Path(plan['prechecks_module'])
        assert sha(hook_path) == plan['source_sha256'][str(hook_path)]
        hook_spec = importlib.util.spec_from_file_location('bell_finite_prechecks', hook_path)
        hook = importlib.util.module_from_spec(hook_spec)
        hook_spec.loader.exec_module(hook)
        module = hook.install(module, plan)
        module.owner_main(plan)
    else:
        assert args.key in trial_keys(plan)
        module.run_driver(plan, args.key)


if __name__ == '__main__':
    main()
