"""Rynn success experiment over the pinned four-GPU owner lifecycle.

The sole new owner manages WM6/7 and CPU-first Rynn4/5, runs one reward gate,
then a full-horizon N64/R1 smoke and continuation of the saved policy CP.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
SOURCE_OWNER = S / 'runs/formal-b16-v1'
THIS = Path(__file__).absolute()
REWARD_FIELDS = {'reward_source', 'rynn_invalid_reward_sentinel', 'rynn_service_urls',
                 'rynn_run_id', 'rynn_batch_size_file', 'rynn_batch_size', 'rynn_timeout_s'}


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


def services_of(plan, kind):
    return [row for row in plan['services'] if row.get('kind') == kind]


def validate_service_layout(plan):
    assert len(plan['services']) == 4
    assert [r['physical_gpu'] for r in services_of(plan, 'wm')] == [6, 7]
    assert [r['physical_gpu'] for r in services_of(plan, 'rynn_success')] == [4, 5]
    assert len({r['key'] for r in plan['services']}) == 4
    assert len({r['url'].rstrip('/') for r in plan['services']}) == 4


def batch_helper(plan):
    path = S / 'formal-b16-control-v1/code/batch16_formal_owner.py'
    assert plan['source_sha256'][str(path)] == sha(path)
    module = load('rynn_pinned_checkpoint_scope_helpers', path)
    # Change only this private module's input anchor, never the old source file.
    module.SOURCE_OWNER = SOURCE_OWNER
    return module


def verify_resume(plan):
    receipt = plan['resume_checkpoint']
    return batch_helper(plan).verify_resume(receipt['receipt'], receipt['sha256'])


def validate_reward_config(cfg, plan):
    train = cfg['env']['train']
    assert train['reward_source'] == 'rynn_success'
    assert train['rynn_invalid_reward_sentinel'] == -1.0
    assert train['rynn_run_id'] and isinstance(train['rynn_run_id'], str)
    assert train['rynn_batch_size_file'] == plan['rm_gate']['output']
    assert train.get('rynn_batch_size', 8) in (4, 8, 16)
    assert train.get('rynn_timeout_s', 7200) > 0
    assert [u.rstrip('/') for u in train['rynn_service_urls']] == [
        row['url'].rstrip('/') for row in services_of(plan, 'rynn_success')]
    assert [u.rstrip('/') for u in train['service_urls']] == [
        row['url'].rstrip('/') for row in services_of(plan, 'wm')]
    assert train['auto_reset'] is False and train['ignore_terminations'] is False
    assert train['use_rel_reward'] is False and train['reward_coef'] == 1.0
    assert train['success_reward_threshold'] == 0.9  # legacy RM diagnostic only


def legacy_shadow(cfg):
    value = copy.deepcopy(cfg)
    train = value['env']['train']
    for key in REWARD_FIELDS:
        train.pop(key, None)
    train['use_rel_reward'] = True
    value['runner']['resume_dir'] = None
    return value


def validate_gate_result(plan):
    path = Path(plan['rm_gate']['output'])
    assert path.is_file() and not path.is_symlink()
    value = read(path)
    assert value.get('passed') is True, 'Rynn batch/semantic gate did not pass'
    assert type(value.get('selected_rm_batch')) is int
    assert value['selected_rm_batch'] in (4, 8, 16)
    return value


def verify_saved_gate(plan):
    gate = Path(plan['owner_dir']) / 'rm_gate/frozen-result.json'
    receipt = read(gate)
    assert receipt['path'] == plan['rm_gate']['output']
    assert receipt['sha256'] == sha(receipt['path']), 'Rynn selected batch evidence changed'
    assert validate_gate_result(plan)['selected_rm_batch'] == receipt['selected_rm_batch']
    return receipt


def service_http(original, rynn_urls, opener=None):
    """Only Rynn control endpoints require a zero-length POST body."""
    known = {value.rstrip('/') for value in rynn_urls}
    opener = opener or urllib.request.urlopen
    def request(url, endpoint='/health', post=False, timeout=5):
        if post and endpoint in ('/onload', '/offload') and url.rstrip('/') in known:
            req = urllib.request.Request(url.rstrip('/') + endpoint, data=b'',
                headers={'Content-Type': 'application/json'})
            with opener(req, timeout=timeout) as response:
                return json.load(response)
        return original(url, endpoint=endpoint, post=post, timeout=timeout)
    return request


def validate_learning_records(records):
    assert records and all(math.isfinite(row['value']) for row in records), 'Missing/nonfinite smoke learning metrics'
    gradients = [row['value'] for row in records if row['tag'].endswith('actor/grad_norm')]
    groups = [row['value'] for row in records if row['tag'].endswith('rynn_effective_group_fraction')]
    assert gradients and any(value > 0 for value in gradients), 'Smoke has no finite nonzero actor gradient'
    assert groups and all(0 <= value <= 1 for value in groups) and any(value > 0 for value in groups), \
        'Smoke has no effective Rynn GRPO group'
    return dict(passed=True, grad_norm=gradients, rynn_effective_group_fraction=groups, records=records)


def smoke_learning_gate(plan):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util.tensor_util import make_ndarray
    row = next(r for r in plan['trials'] if r['key'] == 'startup_smoke')
    cfg = read(row['config'])
    directory = Path(cfg['runner']['logger']['log_path']) / 'tensorboard'
    assert directory.resolve().is_relative_to((Path(plan['owner_dir']) / 'startup_smoke').resolve())
    files, records = [], []
    for path in sorted(directory.rglob('events.out.tfevents.*')):
        stat = path.stat()
        acc = EventAccumulator(str(path), size_guidance={'scalars': 0, 'tensors': 0})
        acc.Reload()
        for kind in ('scalars', 'tensors'):
            for tag in acc.Tags().get(kind, []):
                if not any(key in tag for key in ('rynn_effective_group_fraction', 'actor/grad_norm', 'actor/loss', 'advantage')):
                    continue
                events = acc.Scalars(tag) if kind == 'scalars' else acc.Tensors(tag)
                for event in events:
                    value = event.value if kind == 'scalars' else make_ndarray(event.tensor_proto)
                    records.append(dict(tag=tag, step=event.step, value=float(value)))
        assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns, 'Smoke metric file still changing'
        files.append(dict(path=str(path), bytes=stat.st_size, sha256=sha(path)))
    result = validate_learning_records(records)
    result['files'] = files
    return result


def owner_source_with_gate(source, adoption):
    """Exact, audited extension of the existing launch/catalog transaction."""
    if adoption:
        before = "    assert not (cycle/'rlt-stopped.json').exists(), 'Owner must witness its own borrowing'"
        assert source.count(before) == 1 and source.count('\n    borrowed = False\n') == 1
        source = source.replace(before, '    adopted_borrow = verify_formal_adoption(input_plan)', 1)
        source = source.replace('\n    borrowed = False\n', '\n    borrowed = adopted_borrow\n', 1)
    before = "        for row in plan['trials']:\n            env = dict(base_env)"
    assert source.count(before) == 1
    return source.replace(before,
        "        run_reward_gate(plan, catalog, launch, service_children, base_env)\n" + before, 1)


def install(plan):
    base = Path(plan['base_formal_wrapper'])
    assert plan['source_sha256'][str(base)] == sha(base)
    previous = read(SOURCE_OWNER / 'owner-plan.json')
    assert base == Path(previous['base_formal_wrapper'])
    W = load('rynn_frozen_formal_wrapper', base)
    W.THIS = THIS
    original_formal = W.validate_formal_config
    original_validate = W.validate

    def formal(cfg, row, current, owner, reference):
        validate_reward_config(cfg, current)
        resumed = verify_resume(current)
        assert cfg['runner']['resume_dir'] == resumed['checkpoint_path']
        assert cfg['runner'].get('ckpt_path') is None
        wm_plan = dict(current, services=services_of(current, 'wm'))
        original_formal(legacy_shadow(cfg), row, wm_plan, owner, reference)

    def smoke(cfg, row, current, owner, formal_cfg):
        validate_reward_config(cfg, current)
        assert row['key'] == 'startup_smoke' and row['episode_steps'] == 384 and row['num_envs'] == 64
        assert 0 < row['timeout_seconds'] <= 10800
        runner = cfg['runner']
        assert runner['max_epochs'] == runner['max_steps'] == 1
        assert runner['save_interval'] == 1 and runner['val_check_interval'] == -1
        assert runner.get('resume_dir') is None and runner.get('ckpt_path') is None
        assert cfg['env']['train']['rollout_epoch'] == 1 and cfg['actor']['global_batch_size'] == 768
        restored = copy.deepcopy(cfg)
        for key in ('max_epochs', 'max_steps', 'save_interval', 'val_check_interval', 'resume_dir'):
            restored['runner'][key] = formal_cfg['runner'][key]
        restored['env']['train']['rollout_epoch'] = formal_cfg['env']['train']['rollout_epoch']
        restored['env']['train']['rynn_run_id'] = formal_cfg['env']['train']['rynn_run_id']
        restored['actor']['global_batch_size'] = formal_cfg['actor']['global_batch_size']
        assert W.startup_contract(restored) == W.startup_contract(formal_cfg), 'Smoke changed beyond declared serial budget'

    def validate(current, frozen=False):
        assert current['startup_smoke'] is True and current['wm_batch_size'] == 16
        validate_service_layout(current)
        gate = current['rm_gate']
        assert Path(gate['output']) == Path(current['owner_dir']) / 'rm_gate/result.json'
        assert 0 < gate['timeout_seconds'] <= 7200
        assert isinstance(gate['argv'], list) and all(isinstance(x, str) for x in gate['argv'])
        assert gate['argv'] and Path(gate['argv'][0]).is_absolute()
        assert Path(gate['cwd']).is_absolute()
        assert any(str(path) in gate['argv'] for path in current['source_sha256'] if Path(path).name == 'rynn_gate_client.py')
        for service in services_of(current, 'wm'):
            assert W.M.one_arg(service['argv'], '--wm-batch-size') == '16'
        batch_helper(current).verify_scope_reuse(current)
        return original_validate(current, frozen)

    def reuse_scope(current):
        M = W.M
        assert W.verify_adoption(current)
        assert not M.H.same(read(SOURCE_OWNER / 'owner-identity.json'))
        assert not M.H.gpu_processes([4, 5, 6, 7]), 'Previous owner contexts remain'
        active, fragment, manifest = batch_helper(current).verify_scope_reuse(current)
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
        W.OWNER_ENV.update(fragment)

    W.validate_formal_config, W.validate_startup_smoke = formal, smoke
    W.validate, W.post_borrow = validate, reuse_scope
    M = W.install(plan)
    old_services = M.validate_services

    def typed_services(services, owner):
        validate_service_layout(dict(services=services))
        old_services([r for r in services if r['kind'] == 'wm'], owner)
        # Existing validator's [6,7] assertion is the only changed rule.
        source = __import__('inspect').getsource(old_services)
        before = "    assert [s['physical_gpu'] for s in services] == [6, 7]"
        assert source.count(before) == 1
        source = source.replace(before, "    assert [s['physical_gpu'] for s in services] == [4, 5]", 1)
        before = "        output = owned_path(one_arg(argv, '--output-dir'), exists=False)"
        assert source.count(before) == 1
        source = source.replace(before,
            "        output = owned_path(str(Path(one_arg(argv, '--log-path')).parent), exists=False)", 1)
        namespace = dict(M.__dict__)
        exec(compile(source, str(THIS) + ':rynn-service-validator', 'exec'), namespace)
        namespace['validate_services']([r for r in services if r['kind'] == 'rynn_success'], owner)
        from urllib.parse import urlparse
        assert len({urlparse(r['url']).port for r in services}) == 4, 'Service ports collide'

    M.validate_services = typed_services
    M.http = service_http(M.http, [r['url'] for r in services_of(plan, 'rynn_success')])

    def run_gate(current, catalog, launch, service_children, environment):
        gate = current['rm_gate']
        target = Path(current['owner_dir']) / 'rm_gate'
        target.mkdir(exist_ok=False)
        assert not Path(gate['output']).exists()
        env = dict(environment, CUDA_VISIBLE_DEVICES='')
        for key in M.MASKS[1:]:
            env.pop(key, None)
        child = launch(gate['argv'], gate['cwd'], env, 'rm_gate', target / 'client.log')
        deadline = time.monotonic() + gate['timeout_seconds']
        while child.poll() is None:
            M.resource_snapshot(current, catalog, 'rm_gate')
            assert all(p.poll() is None for p in service_children.values()), 'Service exited during reward gate'
            if time.monotonic() >= deadline:
                raise TimeoutError('Rynn batch gate exceeded declared budget')
            M.atomic(Path(current['owner_dir']) / 'state.json', dict(time=M.H.now(), phase='rm_gate'))
            time.sleep(5)
        M.record(target / 'cleanup.json', M.cleanup(current, catalog, 'rm_gate'))
        assert child.returncode == 0, 'Rynn batch/semantic gate client failed'
        value = validate_gate_result(current)
        for service in services_of(current, 'rynn_success'):
            result = M.http(service['url'], '/offload', post=True, timeout=120)
            assert result['ok'] is True and result['is_offloaded'] is True
        M.record(target / 'frozen-result.json', dict(time=M.H.now(), path=gate['output'],
            sha256=sha(gate['output']), selected_rm_batch=value['selected_rm_batch']))
        M.resource_snapshot(current, catalog, 'rm_gate_complete')

    M.run_reward_gate = run_gate
    donor = Path(plan['base_owner_module']).read_text()
    node = next(n for n in ast.parse(donor).body if isinstance(n, ast.FunctionDef) and n.name == 'owner_main')
    source = ast.get_source_segment(donor, node) + '\n'
    changed = owner_source_with_gate(source, bool(plan.get('borrow_adoption')))
    exec(compile(changed, str(THIS) + ':reward-gated-owner-main', 'exec'), M.__dict__)
    old_record, old_driver = M.record, M.run_driver

    def record(path, value):
        old_record(path, value)
        if Path(path).name == 'owner-plan.json':
            old_record(Path(path).parent / 'rynn-owner-source-delta.json', dict(
                donor_sha256=sha(plan['base_owner_module']), before_sha256=hashlib.sha256(source.encode()).hexdigest(),
                after_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                changes=['reuse audited adoption entry', 'registered reward gate before startup smoke']))
        if Path(path) == Path(plan['owner_dir']) / 'startup_smoke/result.json' and value['exit_code'] == 0:
            wm_plan = dict(plan, services=services_of(plan, 'wm'))
            gate = batch_helper(plan).smoke_batch_gate(wm_plan)
            gate['time'] = M.H.now()
            old_record(Path(plan['owner_dir']) / 'batch16-smoke-gate.json', gate)
            learning = smoke_learning_gate(plan)
            learning['time'] = M.H.now()
            old_record(Path(plan['owner_dir']) / 'rynn-learning-smoke-gate.json', learning)

    def driver(current, key):
        verify_saved_gate(current)
        if key == 'formal':
            target = Path(current['owner_dir']) / 'startup_smoke'
            assert read(target / 'result.json')['exit_code'] == 0
            assert read(target / 'driver-finished.json')['exit_code'] == 0
            assert (target / 'verified-placement.json').is_file()
            batch_helper(current).verify_saved_gate(dict(current, services=services_of(current, 'wm')))
            learning = read(Path(current['owner_dir']) / 'rynn-learning-smoke-gate.json')
            assert learning['passed'] is True
            validate_learning_records(learning['records'])
            for evidence in learning['files']:
                assert sha(evidence['path']) == evidence['sha256'], 'Smoke learning evidence changed'
        return old_driver(current, key)

    M.record, M.run_driver = record, driver
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
