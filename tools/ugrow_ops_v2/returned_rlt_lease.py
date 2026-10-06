"""Single re-borrow of the exact RLT driver dispatched by the v1 U lease.

prepare_from_return -> stop -> verify_release_returned; caller then owns U v2.
restore(stage, terminal_path) retains the v1 terminal/one-dispatch protocol.
No original owner acknowledgements are written. No persistent monitor is added.
The v1 module is loaded separately per call; its on-disk source is never edited.
"""
import copy
from datetime import datetime
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import time

BASE_SHA = '49e0299afe23b7604e33bbe25a2c06301c0d8f241d07cb8c9ef59184e96aa016'
KIND = 'returned_rlt_once_v2'
NO_CP = 'No complete RLT checkpoint; retain RLT until an existing save point'


def _base():
    path = Path(__file__).with_name('rlt_lease.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA
    spec = importlib.util.spec_from_file_location('isolated_v1_lease', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _time(value):
    return float(value) if isinstance(value, (float, int)) else datetime.fromisoformat(value).timestamp()


def _select_checkpoint(base, run, fallback):
    """Fallback only for absent/in-progress saves; a corrupt completed CP fails."""
    base._check_checkpoint(fallback)
    try:
        cp = base._checkpoint(run)
    except RuntimeError as error:
        if str(error) != NO_CP:
            raise
        return copy.deepcopy(fallback)
    assert cp['step'] >= fallback['step'], 'Returned RLT checkpoint regressed'
    return cp


def _bound_base(stage):
    base = _base()
    lease = base._read(Path(stage) / 'lease.json')
    assert lease['lease_kind'] == KIND
    original_checkpoint = base._checkpoint
    target = Path(lease['row']['run'])

    def checkpoint(run):
        if Path(run) != target:
            return original_checkpoint(run)
        fallback = lease['returned_from']['checkpoint']
        base._check_checkpoint(fallback)
        try:
            cp = original_checkpoint(run)
        except RuntimeError as error:
            if str(error) != NO_CP:
                raise
            return copy.deepcopy(fallback)
        assert cp['step'] >= fallback['step'], 'Returned RLT checkpoint regressed'
        return cp

    base._checkpoint = checkpoint
    original_actors = base._actors_and_tree

    def actors_and_tree(current, common, task_plan, ops, identity):
        actors, tree = original_actors(current, common, task_plan, ops, identity)
        assert all(a['job_id'] == current['returned_from']['job_id'] for a in actors), 'Return namespace job changed'
        return actors, tree

    base._actors_and_tree = actors_and_tree
    return base


def prepare_from_return(previous_lease_dir, new_lease_dir, *, gpu, operation_id,
                        new_run, new_namespace):
    """Freeze live return driver + proven parent release + current watch entry."""
    assert gpu in (4, 5) and re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', operation_id)
    assert re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', new_namespace)
    base = _base()
    previous, stage, new_run = Path(previous_lease_dir).resolve(), Path(new_lease_dir).resolve(), Path(new_run)
    assert stage != previous and stage.is_relative_to(Path('/data/chenyiteng'))
    assert new_run.is_absolute() and new_run.resolve().parent == base.ROOT and len(new_run.name.encode()) < 100
    assert not new_run.exists()
    stage.mkdir(parents=True, exist_ok=True)
    # Lock the previous return path too: another caller cannot update its route
    # while this binding is being committed. No running owner's file is edited.
    with base._lock(previous), base._lock(stage):
        assert not (stage / 'lease.json').exists()
        old, common, _, _ = base._context(previous)
        assert old['gpu'] == gpu and old['operation_id'] != operation_id
        assert 'lease_kind' not in old, 'This adapter supports one v1-to-v2 re-borrow'
        release_path = previous / 'rlt-released.json'
        released = base._read(release_path)
        assert released['ready'] is True and released['owner_acknowledged'] is True
        dispatch_path, prepared_path = previous / 'return-dispatched.json', previous / 'return/prepared.json'
        dispatched, prepared = base._read(dispatch_path), base._read(prepared_path)
        assert dispatched['checkpoint'] == prepared['checkpoint'] == released['checkpoint']
        base._check_checkpoint(released['checkpoint'])
        assert all(base._sha(p) == sha for p, sha in prepared['pins'].items())
        attempt_path = previous / 'return-attempt.json'
        attempt = base._read(attempt_path)
        terminal_path = Path(attempt['terminal_receipt'])
        terminal = base._read(terminal_path)
        assert base._sha(terminal_path) == attempt['terminal_sha256']
        assert all(terminal[k] == old[k] for k in ('operation_id', 'gpu', 'gpu_uuid', 'boot_id'))
        assert terminal['decision'] in ('COMPLETE', 'FAILED') and terminal['released'] is True
        original_done = Path(old['original_stage']) / f"finished-{old['task']}-{old['role']}.json"
        assert base._read(original_done)['time'] >= base._read(previous / 'stop-attempt.json')['time']
        task_path = previous / 'return/plan.json'
        task_plan = base._read(task_path)
        row = task_plan['runs'][old['role']]
        assert row['gpus'] == [gpu]
        assert row['run'] == dispatched['run'] == old['new_run']
        assert row['namespace'] == dispatched['namespace'] == old['new_namespace']
        assert str(new_run) != row['run'] and new_namespace != row['namespace']
        runtime = Path(row['run']) / 'runtime'
        identity_path = runtime / 'driver-identity.json'
        identity = base._read(identity_path)
        assert all(identity[k] == dispatched['identity'][k] for k in ('pid', 'uid', 'start'))
        assert identity['namespace'] == row['namespace'] and common.same(identity)
        ops = common.load_module(task_plan['ops'], 'returned_rlt_ops')
        ops.ST = task_path.parent
        ops.checked()
        actors = ops.validate_scoped_actors(task_plan, ops.active_actors(task_plan, row['namespace']))
        jobs = {a['job_id'] for a in actors}
        assert len(jobs) == 1, 'Wait until the returned driver has one identifiable Ray job'
        assert not ops.active_actors(task_plan, new_namespace)
        assert all(not common.same(r['identity']) for r in terminal['runs'])
        assert not any(ops.active_actors(task_plan, r['namespace']) for r in terminal['runs'])
        watch_entry = base._read(old['watch'])['runs'][old['watch_key']]
        watch_proof = previous / 'return-watch-updated.json'
        assert base._read(watch_proof)['target'] == watch_entry
        assert watch_entry['gpus'] == [gpu] and all(watch_entry[k] == row[k] for k in ('run', 'namespace'))
        import yaml
        config = yaml.safe_load((runtime / 'resolved.yaml').read_text())
        assert config['runner']['max_epochs'] == config['runner']['max_steps'] == 3000
        assert config['env']['train']['total_num_envs'] == 8
        assert config['runner']['resume_dir'] == released['checkpoint']['path']
        cp = _select_checkpoint(base, Path(row['run']), released['checkpoint'])
        lease = copy.deepcopy(old)
        lease.update(lease_kind=KIND, operation_id=operation_id, task_plan=str(task_path),
                     row=copy.deepcopy(row), identity=identity, checkpoint=cp,
                     watch_entry=watch_entry, new_run=str(new_run), new_namespace=new_namespace,
                     prepared_at=time.time(), returned_from=dict(previous_lease=str(previous),
                     dispatch=str(dispatch_path), checkpoint=released['checkpoint'],
                     job_id=next(iter(jobs)),
                     original_role_terminal=str(original_done)))
        evidence = (previous / 'lease.json', release_path, previous / 'stop-attempt.json',
                    dispatch_path, prepared_path, attempt_path, terminal_path, original_done,
                    task_path, identity_path, watch_proof, Path(__file__), Path(__file__).with_name('rlt_lease.py'))
        lease['pins'].update({str(p): base._sha(p) for p in evidence})
        lease['pins'].update(prepared['pins'])
        claim = previous / f'returned-gpu{gpu}-lease-claim.json'
        common.save(claim, dict(operation_id=operation_id, lease=str(stage / 'lease.json')), True)
        lease['claim'] = str(claim)
        common.save(stage / 'lease.json', lease, True)
        return lease


def stop(stage):
    """Original exact-driver stop, with this returned run's bounded CP fallback."""
    return _bound_base(stage).stop(Path(stage))


def _release_evidence(base, lease, stopped):
    runtime = Path(lease['row']['run']) / 'runtime'
    finished, cleanup = runtime / 'finished.json', runtime / 'cleanup-targets.json'
    if finished.is_file():
        value = base._read(finished)
        if _time(value['time']) >= stopped['time'] and isinstance(value.get('exit_code'), int):
            return dict(kind='returned_driver_finished', path=str(finished), sha256=base._sha(finished))
    if not cleanup.is_file():
        return None
    value = base._read(cleanup)
    if _time(value['time']) < stopped['time']:
        return None
    jobs = {a['job_id'] for a in stopped['actors']}
    # No guessed job ownership when the return driver was stopped before Ray
    # initialization; leave that exceptional release for explicit review.
    assert len(jobs) == 1 and value['job_id'] in jobs, 'Cleanup job is not the stopped return job'
    assert all(a['job_id'] == value['job_id'] and a['ray_namespace'] == lease['row']['namespace']
               for a in value['actors']), 'Cleanup contains another namespace/job'
    original_done = Path(lease['returned_from']['original_role_terminal'])
    assert base._sha(original_done) == lease['pins'][str(original_done)]
    return dict(kind='returned_driver_cleanup_and_original_role_terminal', path=str(cleanup),
                sha256=base._sha(cleanup), original_role_terminal=str(original_done))


def verify_release_returned(stage):
    """Require this return driver's release evidence and current exact emptiness."""
    stage = Path(stage)
    base = _bound_base(stage)
    with base._lock(stage):
        lease, common, task_plan, ops = base._context(stage)
        stopped = base._read(stage / 'stop-attempt.json')
        actors, tree = base._actors_and_tree(lease, common, task_plan, ops, lease['identity'])
        live = [p for p in stopped['processes'] if common.same(p)]
        evidence = _release_evidence(base, lease, stopped)
        snapshot = common.gpu_snapshot()
        gpu = snapshot[lease['gpu']]
        pids = {p['pid'] for p in live} | set(tree)
        remaining = [r for r in snapshot if any(p['pid'] in pids for p in r['processes'])]
        ready = not actors and not tree and not live and not remaining and evidence is not None and common.empty_target(gpu, lease['gpu_uuid'])
        result = dict(time=time.time(), operation_id=lease['operation_id'], ready=bool(ready),
                      actors=actors, live_processes=live, gpu=gpu, release_evidence=evidence)
        if ready:
            result['checkpoint'] = (base._read(stage / 'rlt-released.json')['checkpoint']
                                    if (stage / 'rlt-released.json').exists()
                                    else base._checkpoint(Path(lease['row']['run'])))
            if not (stage / 'rlt-released.json').exists():
                common.save(stage / 'rlt-released.json', result, True)
        return result


def restore(stage, terminal_receipt):
    """Unchanged terminal decision, one dispatch, watch CAS and first-round proof."""
    return _bound_base(stage).restore(Path(stage), terminal_receipt)
