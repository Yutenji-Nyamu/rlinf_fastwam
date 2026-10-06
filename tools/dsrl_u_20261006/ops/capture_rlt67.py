"""Read-only SZ1 RLT6/7 handoff capture. JSON stdout; no signals or writes.

Run with the deployed training Python using -B. A prior stdout JSON can be
passed through --verify-against to require the same owner, driver identities,
configurations, checkpoints and GPU assignments immediately before review.
This program never authorizes or performs a stop, launch or lease transfer.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
ROOT_POINTER = Path('/data/chenyiteng/deployment-20261004/rlt-after-dojo-n25-v1/active-queue-continuation.json')
EXPECTED_RUNS = {
    6: ('click_bell', 'clean',
        '/data/chenyiteng/results/rlinf-rlt/pi05-rlt-click_bell-clean-n8full-3000-20261002-sz1-after-dojo-n25-20261004-v1-per-card-20261005-v2',
        'per-card-sz1-click_bell-clean-1005-v2'),
    7: ('click_bell', 'combo',
        '/data/chenyiteng/results/rlinf-rlt/rlt-sz1-g7-after-reset4-1006-v9',
        'per-card-sz1-click_bell-combo-1006-v9'),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ident(value):
    return {k: value['start_ticks'] if k == 'start' and 'start' not in value else value[k]
            for k in ('pid', 'uid', 'start')}


def file_record(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def unreused(b, expected):
    current = b.proc(expected['pid'])
    assert current is None or ident(current) == ident(expected), 'Recorded PID was reused'
    return current


def protected_gpu(b, stage, state, row, driver, snapshot):
    """Historical4/5 may finish; newer jobs are inventory, never stop targets."""
    gpu, run = row['gpus'][0], Path(row['run'])
    current = unreused(b, driver)
    role_key = row['task'] + '/' + row['role']
    result = {'gpu': gpu, 'gpu_uuid': snapshot[gpu]['uuid'], 'task': row['task'],
              'role': row['role'], 'run': str(run), 'namespace': row['namespace'],
              'driver': ident(driver)}
    if b.same(driver):
        assert state['roles'][role_key] == 'TRAINING'
        result['historical_state'] = 'RUNNING'
    else:
        assert current is None or current['state'] in ('Z', 'X')
        assert state['roles'][role_key] in ('FAILED', 'COMPLETE')
        files = [stage / ('finished-' + row['task'] + '-' + row['role'] + '.json'),
                 run / 'runtime/finished.json']
        for path in files:
            assert path.is_file(), 'Historical protected role lacks a terminal receipt'
        assert read(files[0])['exit_code'] == read(files[1])['exit_code']
        result.update(historical_state='TERMINAL', terminal_proof=[file_record(path) for path in files])
    return result


def current_context_identities(b, snapshot, gpu):
    result = []
    for item in snapshot[gpu]['processes']:
        current = b.proc(item['pid'])
        if current is None:
            result.append({'type': item['type'], 'pid': item['pid'], 'exited_after_gpu_snapshot': True})
        else:
            result.append({'type': item['type'], 'identity': ident(current)})
    return result


def checkpoint_at(path):
    cp = Path(path)
    step = int(cp.name.removeprefix('global_step_'))
    marker = cp / 'actor/sac_components/rlt_trainer_state/complete.json'
    before = sha(marker)
    state = read(marker)
    assert state.get('complete') is True
    assert state['saved_runner_step'] == step and state['actor_world_size'] == 1
    replay = cp / 'actor/sac_components/replay_buffer/rank_0'
    meta_path, index_path = replay / 'metadata.json', replay / 'trajectory_index.json'
    meta, index = read(meta_path), read(index_path)
    assert meta['total_samples'] == len(index['trajectory_id_list']) > 0
    dcp = cp / 'actor/dcp_checkpoint'
    target = cp / 'actor/sac_components/target_model/checkpoint_rank_0.pt'
    assert (dcp / '.metadata').is_file() and target.is_file() and target.stat().st_size > 0
    assert list(dcp.glob('*.distcp')), 'DCP data shards are absent'
    assert before == sha(marker), 'Checkpoint marker changed during capture'
    return {'path': str(cp), 'step': step, 'update_step': state['update_step'],
            'replay_samples': meta['total_samples'],
            'light_files': [file_record(p) for p in (marker, meta_path, index_path, dcp / '.metadata')],
            'target_model_bytes': target.stat().st_size,
            'dcp_shards': [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(dcp.glob('*.distcp'))]}


def latest_checkpoint(run):
    roots = (run / run.name / 'checkpoints', run / 'checkpoints')
    candidates = sorted({p for root in roots for p in root.glob('global_step_*')
                         if p.name.removeprefix('global_step_').isdigit()},
                        key=lambda p: int(p.name.removeprefix('global_step_')), reverse=True)
    for cp in candidates:
        marker = cp / 'actor/sac_components/rlt_trainer_state/complete.json'
        if not marker.is_file() or read(marker).get('complete') is not True:
            continue
        return checkpoint_at(cp)
    raise AssertionError('No complete RLT checkpoint: ' + str(run))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--expected-owner-pid', required=True, type=int)
    parser.add_argument('--expected-owner-start', required=True, type=int)
    parser.add_argument('--verify-against', type=Path)
    args = parser.parse_args()
    assert os.getuid() == 1003 and socket.gethostname() == 'admin'
    assert Path(read(ROOT_POINTER)['stage']) == args.stage
    plan = read(args.stage / 'plan.json')
    spec = importlib.util.spec_from_file_location('read_only_percard_owner', plan['owner_script'])
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    b = owner.b
    assert b.checked(args.stage) == plan
    state = read(args.stage / 'queue-status.json')
    expected = {'pid': args.expected_owner_pid, 'uid': 1003, 'start': args.expected_owner_start}
    assert ident(state['owner']) == expected and b.same(expected)
    assert not b.same(plan['previous_queue_owner'])
    assert time.time() - float(state['time']) < 120, 'Coordinator heartbeat is stale'
    assert not state.get('errors'), 'Coordinator has unresolved monitor errors'
    assert all(state['roles'][key] == 'TRAINING' for key in ('click_bell/clean', 'click_bell/combo'))
    watch = read(plan['watch'])
    snapshot = b.gpu_snapshot()
    result = {'schema': 1, 'operation': 'READ_ONLY_CAPTURE', 'time': time.time(),
              'stage': str(args.stage), 'owner': expected, 'boot_id': plan['boot_id'],
              'plan': file_record(args.stage / 'plan.json'), 'roles': state['roles'],
              'watch': file_record(plan['watch']), 'gpu_snapshot': snapshot,
              'targets': {}, 'protected': {}, 'protected_contexts': {}}
    for task, ts in plan['tasks'].items():
        q = read(ts['plan'])
        op = b.load_module(q['ops'], 'readonly_ops_' + task)
        op.ST = Path(ts['plan']).parent
        for role in ('clean', 'combo'):
            row = q['runs'][role]
            assert len(row['gpus']) == 1
            gpu, run = row['gpus'][0], Path(row['run'])
            driver = read(run / 'runtime/driver-identity.json')
            assert driver['namespace'] == row['namespace']
            if gpu in (4, 5):
                result['protected'][str(gpu)] = protected_gpu(b, args.stage, state,
                    {**row, 'task': task, 'role': role}, driver, snapshot)
                result['protected_contexts'][str(gpu)] = current_context_identities(b, snapshot, gpu)
                continue
            assert b.same(driver)
            actors = op.active_actors(q, row['namespace'])
            op.validate_scoped_actors(q, actors)
            assert actors, 'Running role has no live actors'
            roots = {driver['pid']} | {a['pid'] for a in actors if a.get('pid')}
            tree = op.process_tree(roots, 1003)
            contexts = {g['gpu']: g['processes'] for g in snapshot if any(x['pid'] in tree for x in g['processes'])}
            assert set(contexts) == {gpu}, 'RLT process has an unexpected GPU context'
            assert all(x['pid'] in tree for x in snapshot[gpu]['processes']), 'Card contains another workload'
            base = {'gpu': gpu, 'gpu_uuid': snapshot[gpu]['uuid'], 'task': task, 'role': role,
                    'run': str(run), 'namespace': row['namespace'], 'driver': ident(driver)}
            assert gpu in EXPECTED_RUNS
            assert (task, role, str(run), row['namespace']) == EXPECTED_RUNS[gpu]
            route = [x for x in watch['runs'].values() if x['gpus'] == [gpu]]
            assert len(route) == 1 and all(route[0][k] == row[k] for k in ('run', 'namespace'))
            cp = latest_checkpoint(run)
            base.update(checkpoint=cp, task_plan=file_record(ts['plan']),
                        config=file_record(run / 'runtime/resolved.yaml'),
                        environment=file_record(run / 'runtime/environment.json'),
                        actors=[{k: a.get(k) for k in ('pid', 'name', 'state', 'ray_namespace', 'job_id', 'actor_id')}
                                for a in actors], process_tree=list(tree.values()))
            result['targets'][str(gpu)] = base
    assert set(result['targets']) == {'6', '7'} and set(result['protected']) == {'4', '5'}
    assert ident(read(args.stage / 'queue-status.json')['owner']) == expected and b.same(expected)
    assert Path(read(ROOT_POINTER)['stage']) == args.stage
    if args.verify_against:
        old = read(args.verify_against)
        for key in ('stage', 'owner', 'boot_id', 'plan', 'protected'):
            assert old[key] == result[key], 'Capture identity changed: ' + key
        for gpu in ('6', '7'):
            for key in ('gpu_uuid', 'task', 'role', 'run', 'namespace', 'driver', 'task_plan', 'config', 'environment'):
                assert old['targets'][gpu][key] == result['targets'][gpu][key], 'Target changed: ' + gpu + '/' + key
            previous_cp = checkpoint_at(old['targets'][gpu]['checkpoint']['path'])
            assert previous_cp == old['targets'][gpu]['checkpoint'], 'Previously selected checkpoint changed'
        result['verified_against'] = file_record(args.verify_against)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
