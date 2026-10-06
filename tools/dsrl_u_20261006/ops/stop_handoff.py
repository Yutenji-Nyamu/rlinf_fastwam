"""Review draft: stop only the already-dispatched RLT6/7 drivers.

Default is a read-only dry run. Execution requires --execute-stop-rlt67.
It never signals the shared coordinator or GPU4/5, starts DSRL/RLT, performs
GPU reset, or escalates to process-tree killing. The first transaction is
single use; after an error, inspect its evidence instead of replaying it.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from capture_rlt67 import checkpoint_at, current_context_identities, file_record, ident, read, sha, unreused


def save_exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_capture(args):
    assert sha(args.capture) == args.capture_sha256, 'Unexpected capture file'
    old = read(args.capture)
    script = Path(__file__).with_name('capture_rlt67.py')
    argv = [sys.executable, '-B', str(script), '--stage', old['stage'],
            '--expected-owner-pid', str(old['owner']['pid']),
            '--expected-owner-start', str(old['owner']['start']),
            '--verify-against', str(args.capture)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, 'Read-only revalidation failed: ' + result.stderr[-4000:]
    value = json.loads(result.stdout)
    assert set(value['targets']) == {'6', '7'}
    return value


def load_live_modules(capture):
    stage = Path(capture['stage'])
    plan = read(stage / 'plan.json')
    spec = importlib.util.spec_from_file_location('handoff_existing_owner', plan['owner_script'])
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    b = owner.b
    assert b.checked(stage) == plan
    modules = {}
    for gpu, target in capture['targets'].items():
        task_spec = plan['tasks'][target['task']]
        q = read(task_spec['plan'])
        op = b.load_module(q['ops'], 'handoff_existing_ops_' + gpu)
        op.ST = Path(task_spec['plan']).parent
        row = q['runs'][target['role']]
        assert row['gpus'] == [int(gpu)] and row['run'] == target['run']
        assert row['namespace'] == target['namespace']
        modules[gpu] = {'ops': op, 'task_plan': q, 'row': row}
    return b, plan, modules


def protect(b, capture, allow_owner_terminal=False):
    unreused(b, capture['owner'])
    state = read(Path(capture['stage']) / 'queue-status.json')
    assert ident(state['owner']) == capture['owner']
    if b.same(capture['owner']):
        assert time.time() - state['time'] < 120, 'Original coordinator heartbeat is stale'
    else:
        assert allow_owner_terminal, 'Original coordinator exited before stopping targets'
        final_path = Path(capture['stage']) / 'queue-finished.json'
        assert final_path.is_file(), 'Original coordinator lacks a natural final receipt'
        final = read(final_path)
        assert ident(final['owner']) == capture['owner']
        assert final['roles'] == state['roles'] and not final.get('retained_drivers')
        assert all(value in ('FAILED', 'COMPLETE') for value in final['roles'].values())
        assert not final.get('error'), 'Original coordinator ended with an unhandled exception'
    for gpu, row in capture['protected'].items():
        unreused(b, row['driver'])
        assert ident(read(Path(row['run']) / 'runtime/driver-identity.json')) == row['driver']
        if row.get('historical_state', 'RUNNING') == 'TERMINAL':
            assert not b.same(row['driver'])
            assert state['roles'][row['task'] + '/' + row['role']] in ('FAILED', 'COMPLETE')
            for proof in row['terminal_proof']:
                assert sha(proof['path']) == proof['sha256']
        else:
            assert b.same(row['driver']), 'Previously running protected driver exited; refresh before action'
            assert state['roles'][row['task'] + '/' + row['role']] == 'TRAINING'


def release_snapshot(b, modules, capture):
    protect(b, capture, allow_owner_terminal=True)
    snapshot = b.gpu_snapshot()
    state = read(Path(capture['stage']) / 'queue-status.json')
    detail = {}
    for gpu, row in capture['targets'].items():
        source = modules[gpu]
        actors = source['ops'].active_actors(source['task_plan'], row['namespace'])
        source['ops'].validate_scoped_actors(source['task_plan'], actors)
        live_tree = [item for item in row['process_tree'] if b.same(item)]
        queue_role = state['roles'][row['task'] + '/' + row['role']]
        detail[gpu] = {'driver_alive': b.same(row['driver']),
                       'actors': [{k: item.get(k) for k in ('pid', 'name', 'state', 'job_id', 'actor_id')}
                                  for item in actors], 'recorded_processes_alive': live_tree,
                       'gpu': snapshot[int(gpu)], 'queue_role': queue_role,
                       'queue_finished': read(Path(capture['stage']) / ('finished-' + row['task'] + '-' + row['role'] + '.json'))
                       if (Path(capture['stage']) / ('finished-' + row['task'] + '-' + row['role'] + '.json')).is_file() else None}
        detail[gpu]['released'] = (not detail[gpu]['driver_alive'] and not actors and not live_tree
                                   and b.empty_target(snapshot[int(gpu)], row['gpu_uuid'])
                                   and queue_role in ('FAILED', 'COMPLETE')
                                   and detail[gpu]['queue_finished'] is not None)
    return {'time': time.time(), 'targets': detail,
            'protected': capture['protected'], 'owner': capture['owner'],
            'protected_contexts': {str(g): current_context_identities(b, snapshot, g) for g in (4, 5)},
            'original_owner_alive': b.same(capture['owner']),
            'all_released': all(item['released'] for item in detail.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--operation-dir', required=True, type=Path)
    parser.add_argument('--execute-stop-rlt67', action='store_true')
    args = parser.parse_args()
    # An exact new transaction is required. This draft has no retry/overwrite mode.
    args.operation_dir = args.operation_dir.resolve()
    assert args.operation_dir.parent == Path('/data/chenyiteng/deployment-20261006')
    assert args.operation_dir.name.startswith('dsrl-') and not args.operation_dir.exists()
    capture = verify_capture(args)
    b, plan, modules = load_live_modules(capture)
    protect(b, capture)
    if not args.execute_stop_rlt67:
        print(json.dumps({'operation': 'READ_ONLY_STOP_REVIEW', 'ready': True,
                          'targets': capture['targets'], 'protected': capture['protected'],
                          'capture': file_record(args.capture),
                          'stop_script': file_record(__file__), 'will_signal': [6, 7],
                          'will_not_signal': ['coordinator', 4, 5],
                          'operation_dir': str(args.operation_dir)}))
        return
    args.operation_dir.mkdir(mode=0o700)
    lock = (args.operation_dir / 'operation.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    own = b.proc(os.getpid())
    attempt = {'time': time.time(), 'operation_owner': own,
               'reason': 'User-authorized DSRL Clean6/U7 priority; preserve RLT4/5',
               'capture': capture, 'capture_file': file_record(args.capture),
               'stop_script': file_record(__file__), 'signals': 'existing signal_exact_driver only',
               'handoff_contract': 'No RLT fallback during development/smoke/formal; separate lease owner required before DSRL launch'}
    save_exclusive(args.operation_dir / 'stop-attempt.json', attempt)
    try:
        # Checkpoint proof is refreshed immediately before the first irreversible signal.
        for row in capture['targets'].values():
            assert checkpoint_at(row['checkpoint']['path']) == row['checkpoint']
        protect(b, capture)
        for gpu in ('6', '7'):
            row, source = capture['targets'][gpu], modules[gpu]
            assert b.same(row['driver']), 'Target driver changed before signal'
            b.signal_exact_driver(source['ops'], source['row'], row['driver'],
                                  'DSRL67_priority_20261006', args.operation_dir)
        deadline = time.monotonic() + 300
        last = None
        while True:
            last = release_snapshot(b, modules, capture)
            b.save(args.operation_dir / 'release-status.json', last)
            if last['all_released']:
                # CP files must survive cleanup; leave exact inputs for the future resume.
                for row in capture['targets'].values():
                    assert checkpoint_at(row['checkpoint']['path']) == row['checkpoint']
                save_exclusive(args.operation_dir / 'rlt67-released.json', last)
                print(json.dumps({'operation': 'RLT67_RELEASED', 'receipt': str(args.operation_dir / 'rlt67-released.json'),
                                  'operation_owner': own, 'checkpoints': {g: row['checkpoint'] for g, row in capture['targets'].items()}}))
                return
            assert time.monotonic() < deadline, 'Scoped release exceeded 300 seconds; no kill escalation performed'
            time.sleep(2)
    except BaseException:
        failure = {'time': time.time(), 'operation_owner': own,
                   'error': traceback.format_exc(), 'instruction': 'Inspect this transaction; do not replay stop or launch'}
        save_exclusive(args.operation_dir / 'stop-failed.json', failure)
        raise


if __name__ == '__main__':
    main()
