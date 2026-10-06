"""One-card SZ1 RLT lease; caller owns the U smoke/formal lifetime.

No CLI, polling, guardian, Ray shutdown, reset, or non-target process signals.
Call prepare -> stop -> verify_release until ready. Only after the U transaction
has ended call restore with a caller-written terminal receipt. Repeated restore
calls inspect the one dispatched driver and eventually verify its first round.

Terminal receipt matches runtime.py: operation_id, gpu, gpu_uuid, boot_id,
decision ('COMPLETE' or 'FAILED'), released=True, runs (identity/namespace/run),
and scope. Each run may also carry the process identities observed while running.
A caller that failed before launching U supplies an empty runs list. A smoke exit
is not terminal; only the caller's whole-transaction terminal.json is accepted.
"""
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time

POINTER = Path('/data/chenyiteng/deployment-20261004/rlt-after-dojo-n25-v1/active-queue-continuation.json')
# p002 inventory: this owner marks a dispatched role done once, without respawn.
OWNER_SHA = '56f6e0ac152033e753e89dd2b14d6d29d422a06711f4f6f0452aee110fd5ca00'
COMMON_SHA = '8e21677d0e3b0d58b810edab574d2903a54161be41cadff716b23f8f96d74029'
ROOT = Path('/data/chenyiteng/results/rlinf-rlt')
OUTPUT_KEYS = {
    'runner.logger.log_path', 'runner.logger.experiment_name',
    'runner.per_worker_log_path', 'runner.resume_dir',
    'env.train.task_config.save_path', 'env.eval.task_config.save_path',
    'env.train.video_cfg.video_base_dir', 'env.eval.video_cfg.video_base_dir',
}


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lock(stage):
    handle = (stage / 'operation.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _common(plan):
    owner = Path(plan['owner_script'])
    assert _sha(owner) == plan['pins'][str(owner)] == OWNER_SHA
    base = next(ast.literal_eval(n.value) for n in ast.parse(owner.read_text()).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BASE' for t in n.targets))
    assert _sha(base) == plan['pins'][base] == COMMON_SHA
    return _load_module(base, 'ugrow_rlt_common')


def _checkpoint(run):
    """Same complete-marker/replay gate as the established cutover helper."""
    paths = sorted((run / run.name / 'checkpoints').glob('global_step_*'),
                   key=lambda p: int(p.name.rsplit('_', 1)[1]), reverse=True)
    for cp in paths:
        marker = cp / 'actor/sac_components/rlt_trainer_state/complete.json'
        if not marker.is_file() or _read(marker).get('complete') is not True:
            continue
        state = _read(marker)
        step = int(cp.name.rsplit('_', 1)[1])
        assert state['saved_runner_step'] == step and state['actor_world_size'] == 1
        replay = cp / 'actor/sac_components/replay_buffer/rank_0'
        meta, index = _read(replay / 'metadata.json'), _read(replay / 'trajectory_index.json')
        assert meta['total_samples'] == len(index['trajectory_id_list']) > 0
        dcp = cp / 'actor/dcp_checkpoint/.metadata'
        target = cp / 'actor/sac_components/target_model/checkpoint_rank_0.pt'
        assert dcp.is_file() and target.stat().st_size > 0
        return dict(path=str(cp), step=step, update_step=state['update_step'],
                    replay_samples=meta['total_samples'],
                    pins={str(p): _sha(p) for p in (marker, replay / 'metadata.json', replay / 'trajectory_index.json', dcp)},
                    target=dict(path=str(target), bytes=target.stat().st_size, mtime_ns=target.stat().st_mtime_ns))
    raise RuntimeError('No complete RLT checkpoint; retain RLT until an existing save point')


def _check_checkpoint(cp):
    assert all(_sha(p) == digest for p, digest in cp['pins'].items())
    target = Path(cp['target']['path']).stat()
    assert target.st_size == cp['target']['bytes'] and target.st_mtime_ns == cp['target']['mtime_ns']


def _context(stage):
    lease = _read(stage / 'lease.json')
    assert lease['gpu'] in (4, 5) and os.getuid() == lease['uid'] == 1003
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == lease['boot_id']
    assert all(_sha(p) == digest for p, digest in lease['pins'].items()), 'Frozen lease source changed'
    plan = _read(Path(lease['original_stage']) / 'plan.json')
    common = _common(plan)
    task_plan = _read(lease['task_plan'])
    ops = common.load_module(task_plan['ops'], 'ugrow_rlt_ops')
    ops.ST = Path(lease['task_plan']).parent
    ops.checked()
    return lease, common, task_plan, ops


def _actors_and_tree(lease, common, task_plan, ops, identity):
    actors = ops.active_actors(task_plan, lease['row']['namespace'])
    jobs = {a['job_id'] for a in actors}
    assert len(jobs) <= 1, 'Multiple Ray jobs in target namespace'
    ops.validate_scoped_actors(task_plan, actors)
    roots = {a['pid'] for a in actors if a.get('pid')}
    if common.same(identity):
        roots.add(identity['pid'])
    return actors, ops.process_tree(roots, lease['uid'])


def prepare(stage, *, gpu, operation_id, new_run, new_namespace, pointer=POINTER):
    """Bind the currently dispatched target; only create this lease's records."""
    assert gpu in (4, 5) and re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', operation_id)
    assert re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', new_namespace)
    stage, pointer, new_run = Path(stage).resolve(), Path(pointer), Path(new_run)
    assert stage.is_relative_to(Path('/data/chenyiteng'))
    assert new_run.is_absolute() and new_run.resolve().parent == ROOT and len(new_run.name.encode()) < 100
    assert not new_run.exists()
    stage.mkdir(parents=True, exist_ok=True)
    with _lock(stage):
        assert not (stage / 'lease.json').exists(), 'Lease already prepared'
        original = Path(_read(pointer)['stage'])
        plan = _read(original / 'plan.json')
        common = _common(plan)
        common.checked(original)
        matches = []
        for task, item in plan['tasks'].items():
            task_plan = _read(item['plan'])
            matches += [(task, role, item['plan'], task_plan, row) for role, row in task_plan['runs'].items()
                        if role in ('clean', 'combo') and row['gpus'] == [gpu]]
        assert len(matches) == 1
        task, role, task_path, task_plan, row = matches[0]
        runtime = Path(row['run']) / 'runtime'
        identity = _read(runtime / 'driver-identity.json')
        dispatched_path = original / f'dispatched-{task}-{role}.json'
        dispatched = _read(dispatched_path)
        assert all(dispatched['identity'][k] == identity[k] for k in ('pid', 'uid', 'start'))
        assert identity['namespace'] == row['namespace'] and common.same(identity)
        assert dispatched['run'] == row['run'] and dispatched['namespace'] == row['namespace']
        assert new_namespace != row['namespace'] and str(new_run) != row['run']
        watch = _read(plan['watch'])
        entries = [(k, v) for k, v in watch['runs'].items() if v['gpus'] == [gpu]]
        assert len(entries) == 1
        key, entry = entries[0]
        assert all(entry[k] == row[k] for k in ('run', 'namespace'))
        cp = _checkpoint(Path(row['run']))
        pins = {str(p): _sha(p) for p in (original / 'plan.json', Path(task_path), dispatched_path,
                runtime / 'resolved.yaml', runtime / 'environment.json', Path(task_plan['ops']))}
        # Other windows may normally hand off GPU6/7. Pin this card's scope and
        # RLT source, not unrelated profiles, old runs, or their live routes.
        env = _read(runtime / 'environment.json')
        manifest = Path(env['RLINF_OPENDW_GPU_SCOPE_MANIFEST'])
        scope = _read(manifest)
        assert scope['physical_gpu'] == gpu and scope['gpu_uuid'] == plan['dojo'][str(gpu)]['uuid']
        for path in (plan['owner_script'], common.__file__, str(manifest)):
            assert _sha(path) == plan['pins'][path]
            pins[path] = plan['pins'][path]
        for kind in ('marker', 'profile', 'bootstrap', 'runtime'):
            path, digest = scope[kind + '_path'], scope[kind + '_sha256']
            assert _sha(path) == digest
            pins[path] = digest
        lease = dict(operation_id=operation_id, gpu=gpu, gpu_uuid=plan['dojo'][str(gpu)]['uuid'],
                     uid=1003, boot_id=plan['boot_id'], pointer=str(pointer), original_stage=str(original),
                     original_owner=_read(original / 'owner-identity.json'), task=task, role=role,
                     task_plan=task_path, row=row, identity=identity, checkpoint=cp, pins=pins,
                     watch=plan['watch'], watch_key=key, watch_entry=entry,
                     new_run=str(new_run), new_namespace=new_namespace, prepared_at=time.time())
        ops = common.load_module(task_plan['ops'], 'ugrow_prepare_ops')
        ops.ST = Path(task_path).parent
        assert not ops.active_actors(task_plan, new_namespace)
        claim = original / f'ugrow-gpu{gpu}-lease.json'
        common.save(claim, dict(operation_id=operation_id, lease=str(stage / 'lease.json')), True)
        lease['claim'] = str(claim)
        common.save(stage / 'lease.json', lease, True)
        return lease


def stop(stage):
    """Signal only this lease's driver; its original finally cleans its job."""
    stage = Path(stage)
    with _lock(stage):
        lease, common, task_plan, ops = _context(stage)
        attempt = stage / 'stop-attempt.json'
        if attempt.exists():
            return _read(attempt)  # A failed/ambiguous attempt is never re-signalled.
        assert _read(lease['claim']) == dict(operation_id=lease['operation_id'], lease=str(stage / 'lease.json'))
        assert _read(lease['watch'])['runs'][lease['watch_key']] == lease['watch_entry'], 'Target GPU route changed'
        assert common.same(lease['identity'])
        argv = (Path('/proc') / str(lease['identity']['pid']) / 'cmdline').read_bytes().split(b'\0')
        assert os.fsencode(str(Path(lease['task_plan']).parent)) in argv
        assert argv[argv.index(b'driver') + 1] == os.fsencode(lease['role'])
        actors, tree = _actors_and_tree(lease, common, task_plan, ops, lease['identity'])
        snapshot = common.gpu_snapshot()
        assert all(p['pid'] in tree for p in snapshot[lease['gpu']]['processes']), 'Unrelated target GPU process'
        assert snapshot[lease['gpu']]['uuid'] == lease['gpu_uuid']
        assert not any(p['pid'] in tree for r in snapshot if r['gpu'] != lease['gpu'] for p in r['processes'])
        record = dict(time=time.time(), operation_id=lease['operation_id'], identity=lease['identity'],
                      processes=list(tree.values()), actors=actors, checkpoint=_checkpoint(Path(lease['row']['run'])),
                      signal_set=[lease['identity']], gpu=snapshot[lease['gpu']])
        common.save(attempt, record, True)
        common.signal_exact_driver(ops, lease['row'], lease['identity'], 'ugrow_priority_gpu' + str(lease['gpu']), stage)
        return record


def verify_release(stage):
    """One snapshot; return ready=False while original driver cleanup is pending."""
    stage = Path(stage)
    with _lock(stage):
        lease, common, task_plan, ops = _context(stage)
        stopped = _read(stage / 'stop-attempt.json')
        actors, tree = _actors_and_tree(lease, common, task_plan, ops, lease['identity'])
        live = [p for p in stopped['processes'] if common.same(p)]
        gpu = common.gpu_snapshot()[lease['gpu']]
        finished = Path(lease['original_stage']) / f"finished-{lease['task']}-{lease['role']}.json"
        # The frozen owner cannot respawn after this receipt; natural owner exit is allowed.
        acknowledged = finished.is_file() and _read(finished)['time'] >= stopped['time']
        queue_final = Path(lease['original_stage']) / 'queue-finished.json'
        if not acknowledged and queue_final.is_file() and not common.same(lease['original_owner']):
            final = _read(queue_final)
            key = lease['task'] + '/' + lease['role']
            acknowledged = (final['time'] >= stopped['time'] and
                            final['roles'][key] in ('COMPLETE', 'FAILED') and
                            key not in final.get('retained_drivers', {}))
        ready = not actors and not tree and not live and acknowledged and common.empty_target(gpu, lease['gpu_uuid'])
        result = dict(time=time.time(), operation_id=lease['operation_id'], ready=bool(ready),
                      actors=actors, live_processes=live, gpu=gpu, owner_acknowledged=acknowledged)
        if ready:
            result['checkpoint'] = (_read(stage / 'rlt-released.json')['checkpoint']
                                    if (stage / 'rlt-released.json').exists()
                                    else _checkpoint(Path(lease['row']['run'])))
        if ready and not (stage / 'rlt-released.json').exists():
            cleanup = Path(lease['row']['run']) / 'runtime/cleanup-targets.json'
            result['cleanup_receipt'] = dict(path=str(cleanup), sha256=_sha(cleanup)) if cleanup.is_file() else None
            common.save(stage / 'rlt-released.json', result, True)
        return result


def _flatten(value, prefix=''):
    if isinstance(value, dict):
        return {k2: v2 for k, v in value.items() for k2, v2 in _flatten(v, prefix + ('.' if prefix else '') + k).items()}
    return {prefix: value}


def _prepare_return(stage, lease, common, task_plan):
    import yaml
    cp = _read(stage / 'rlt-released.json')['checkpoint']
    _check_checkpoint(cp)
    old, new = Path(lease['row']['run']), Path(lease['new_run'])
    assert not new.exists()
    cfg = yaml.safe_load((old / 'runtime/resolved.yaml').read_text())
    original = copy.deepcopy(cfg)

    def replace(v):
        if isinstance(v, dict):
            return {k: replace(x) for k, x in v.items()}
        if isinstance(v, list):
            return [replace(x) for x in v]
        if isinstance(v, str):
            return new.name if v == old.name else v.replace(str(old), str(new))
        return v

    cfg = replace(cfg)
    cfg['runner']['resume_dir'] = cp['path']
    changes = {k: [v, _flatten(cfg)[k]] for k, v in _flatten(original).items() if v != _flatten(cfg)[k]}
    assert set(changes) <= OUTPUT_KEYS, changes
    assert cfg['runner']['max_epochs'] == cfg['runner']['max_steps'] == 3000
    assert cfg['env']['train']['total_num_envs'] == 8
    env = _read(old / 'runtime/environment.json')
    env['RLT_LOG_ROOT'] = str(new)
    if 'CLUSTER_NAMESPACE' in env:
        assert env['CLUSTER_NAMESPACE'] == lease['row']['namespace']
        env['CLUSTER_NAMESPACE'] = lease['new_namespace']
    q = copy.deepcopy(task_plan)
    q['runs'][lease['role']].update(run=str(new), namespace=lease['new_namespace'])
    ret = stage / 'return'
    ret.mkdir()
    runtime = new / 'runtime'
    runtime.mkdir(parents=True)
    (runtime / 'resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    common.save(runtime / 'environment.json', env, True)
    (runtime / 'environment.json').chmod(0o600)
    common.save(ret / 'plan.json', q, True)
    common.save(ret / 'prepared.json', dict(checkpoint=cp, config_changes=changes,
                pins={str(p): _sha(p) for p in (ret / 'plan.json', runtime / 'resolved.yaml', runtime / 'environment.json')}), True)
    return ret, q


def _update_watch(stage, lease, common):
    watch = Path(lease['watch'])
    with (watch.parent / 'route.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = _read(watch)
        after = copy.deepcopy(before)
        expected = copy.deepcopy(lease['watch_entry'])
        expected.update(run=lease['new_run'], namespace=lease['new_namespace'])
        current = before['runs'][lease['watch_key']]
        if current == expected:
            return
        assert current == lease['watch_entry'], 'This GPU watch route changed; do not overwrite'
        after['runs'][lease['watch_key']] = expected
        common.save(stage / 'return-watch-before.json', before, True)
        common.save(watch, after)
        assert _read(watch) == after  # Every other entry is preserved under route.lock.
        common.save(stage / 'return-watch-updated.json', dict(time=time.time(), key=lease['watch_key'], target=expected), True)


def restore(stage, terminal_receipt):
    """Dispatch once after U terminal release; repeat only to inspect acceptance."""
    stage = Path(stage)
    with _lock(stage):
        lease, common, task_plan, ops = _context(stage)
        receipt = _read(terminal_receipt)
        assert all(receipt[k] == lease[k] for k in ('operation_id', 'gpu', 'gpu_uuid', 'boot_id'))
        assert Path(terminal_receipt).name == 'terminal.json'
        assert receipt['released'] is True and receipt['decision'] in ('COMPLETE', 'FAILED')
        assert isinstance(receipt['runs'], list)
        processes = [r['identity'] for r in receipt['runs']]
        processes += [p for r in receipt['runs'] for p in r.get('processes', [])]
        namespaces = [r['namespace'] for r in receipt['runs']]
        assert all(p['uid'] == lease['uid'] and not common.same(p) for p in processes)
        assert len(namespaces) == len(set(namespaces))
        assert all(isinstance(ns, str) and ns and ns not in (lease['row']['namespace'], lease['new_namespace']) for ns in namespaces)
        assert not any(ops.active_actors(task_plan, ns) for ns in namespaces)
        if receipt['runs']:
            scope = receipt['scope']
            assert scope['released'] is True and not scope['actors'] and not scope['tree'] and not scope['outside']
            assert common.empty_target(scope['gpu'], lease['gpu_uuid'])
        assert (stage / 'rlt-released.json').is_file()
        dispatched = stage / 'return-dispatched.json'
        if not dispatched.exists():
            assert not (stage / 'return-attempt.json').exists(), 'Ambiguous return attempt; inspect before recovery'
            assert not common.same(lease['identity']) and not ops.active_actors(task_plan, lease['row']['namespace'])
            assert common.empty_target(common.gpu_snapshot()[lease['gpu']], lease['gpu_uuid'])
            assert not ops.active_actors(task_plan, lease['new_namespace'])
            current = _read(lease['watch'])['runs'][lease['watch_key']]
            assert current == lease['watch_entry'], 'This GPU was reassigned'
            ret, q = _prepare_return(stage, lease, common, task_plan)
            ops.ST = ret
            ops.checked()
            common.save(stage / 'return-attempt.json', dict(time=time.time(), terminal_receipt=str(terminal_receipt),
                        terminal_sha256=_sha(terminal_receipt), operation_id=lease['operation_id']), True)
            started = time.time()
            child = ops.launch(lease['role'])
            common.save(dispatched, dict(time=started, identity=common.proc(child.pid),
                        run=lease['new_run'], namespace=lease['new_namespace'], checkpoint=_read(ret / 'prepared.json')['checkpoint']), True)
        dispatch = _read(dispatched)
        _update_watch(stage, lease, common)
        row = {**lease['row'], 'run': lease['new_run'], 'namespace': lease['new_namespace']}
        actors = ops.active_actors(task_plan, row['namespace'])
        ops.validate_scoped_actors(task_plan, actors)
        roots = {a['pid'] for a in actors if a.get('pid')}
        if common.same(dispatch['identity']):
            roots.add(dispatch['identity']['pid'])
        tree = ops.process_tree(roots, lease['uid'])
        snapshot = common.gpu_snapshot()
        outside = [r for r in snapshot if r['gpu'] != lease['gpu'] and any(p['pid'] in tree for p in r['processes'])]
        assert not outside, 'Returned RLT has contexts outside its assigned card'
        proof = common.first_round(row, dispatch['time']) if common.same(dispatch['identity']) else None
        growth = proof and any(v['value'] > dispatch['checkpoint']['replay_samples'] for k, v in proof.items() if 'global_min_replay_size' in k)
        advanced = proof and proof['env/success_once']['step'] >= dispatch['checkpoint']['step']
        result = dict(time=time.time(), operation_id=lease['operation_id'], dispatched=True,
                      identity=dispatch['identity'], alive=common.same(dispatch['identity']),
                      first_round_verified=bool(actors and growth and advanced), metrics=proof,
                      gpu=snapshot[lease['gpu']], run=row['run'], namespace=row['namespace'])
        if result['first_round_verified'] and not (stage / 'rlt-returned.json').exists():
            common.save(stage / 'rlt-returned.json', result, True)
        return result
