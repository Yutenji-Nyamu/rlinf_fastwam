"""Unique two-card lease; probes/smokes retain it, formal terminal returns RLT.

No automatic restarts, GPU reset, shared-Ray shutdown, or broad process kill.
Use submit to create immutable requests after preparing each runtime/config.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from lease_common import checked, environment, gpu_released, original_protected, read, save, sha


def request_checked(p, req):
    assert req['gpu'] in (6, 7) and req['kind'] in ('probe', 'smoke', 'formal')
    assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{2,79}', req['request_id'])
    assert isinstance(req['pins'], dict) and req['pins']
    repo = Path(p['dsrl_repo']).resolve()
    entry = (repo / req['entry']).resolve()
    assert entry.is_relative_to(repo) and entry.is_file()
    assert str(entry) in req['pins'], 'Entry must be pinned'
    for path, digest in req['pins'].items():
        assert sha(path) == digest, 'Request source/config changed: ' + path
    runtime = Path(req['runtime']).resolve()
    assert runtime.is_relative_to(Path('/data/chenyiteng')) and runtime.is_dir()
    assert not (runtime / 'driver-identity.json').exists() and not (runtime / 'driver.log').exists()
    assert req.get('env', {}) == {}, 'Child environment comes from the reviewed lease factory'
    if req['kind'] != 'probe':
        assert req['entry'] == 'examples/embodiment/train_embodied_agent.py'
        assert req['namespace'].startswith('dsrl-u-20261006-')
        assert str(runtime / 'resolved.yaml') in req['pins']
    else:
        assert req.get('namespace') is None
        assert all(isinstance(arg, str) for arg in req.get('args', []))
    if 'repo_head' in req:
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == req['repo_head']
    return runtime


def submit(control, request):
    p, b, op = checked(control)
    state = read(control / 'status.json')
    assert b.same(state['owner']) and time.time() - state['time'] < 120
    req = read(request)
    request_checked(p, req)
    assert state['slots'][str(req['gpu'])]['state'] == 'DEV_HOLD'
    # Atomic rename prevents the owner from reading a partially written request.
    target = control / 'requests' / (req['request_id'] + '.json')
    with (control / 'requests.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not target.exists() and not (control / 'receipts' / (req['request_id'] + '-claimed.json')).exists()
        for pending in (control / 'requests').glob('*.json'):
            other = read(pending)
            if other['gpu'] == req['gpu']:
                assert (control / 'receipts' / (other['request_id'] + '-claimed.json')).exists(), 'Card already has a pending request'
        save(target, req)
    print(__import__('json').dumps({'submitted': str(target), 'sha256': sha(target)}))


def tree_snapshot(p, b, op, child, req):
    actors = op.active_actors(p, req['namespace']) if req.get('namespace') else []
    op.validate_scoped_actors(p, actors)
    roots = {child['identity']['pid']} if b.same(child['identity']) else set()
    roots |= {item['pid'] for item in actors if item.get('pid')}
    tree = op.process_tree(roots, p['uid'])
    child['identities'].update({str(pid): value for pid, value in tree.items()})
    actual = b.gpu_snapshot()
    outside = [row for row in actual if row['gpu'] != req['gpu'] and any(x['pid'] in tree for x in row['processes'])]
    return actors, tree, actual, outside


def update_watch(p, gpu):
    path = Path(p['watch'])
    slot = p['slots'][gpu]
    with (path.parent / 'route.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = read(path)
        new = copy.deepcopy(old)
        pairs = [(key, row) for key, row in new['runs'].items() if row['gpus'] == [int(gpu)]]
        assert len(pairs) == 1
        key, row = pairs[0]
        assert all(row[k] == slot['original'][k] for k in ('run', 'namespace'))
        row.update(run=slot['fallback']['run'], namespace=slot['fallback']['namespace'])
        if 'task' in row:
            row['task'] = slot['original']['task']
        save(path, new)
        assert all(new['runs'][k] == old['runs'][k] for k in old['runs'] if k != key)


def restore(p, b, op, control, gpu, state):
    slot = p['slots'][gpu]
    released, snapshot, actors = gpu_released(p, b, op, gpu, state.get('namespace'))
    assert released, 'DSRL not fully released'
    assert not (control / 'receipts' / ('gpu' + gpu + '-restore-attempt.json')).exists()
    original_protected(p, b)
    cp = Path(slot['original']['checkpoint']['path'])
    for record in slot['original']['checkpoint']['light_files']:
        assert sha(record['path']) == record['sha256'], 'RLT checkpoint metadata changed'
    assert (cp / 'actor/sac_components/target_model/checkpoint_rank_0.pt').stat().st_size == slot['original']['checkpoint']['target_model_bytes']
    for shard in slot['original']['checkpoint']['dcp_shards']:
        assert (cp / 'actor/dcp_checkpoint' / shard['name']).stat().st_size == shard['bytes']
    op.ST = Path(p['fallback_plan']).parent
    plan = op.checked()
    assert plan['runs'][slot['fallback_role']] == slot['fallback']
    save(control / 'receipts' / ('gpu' + gpu + '-restore-attempt.json'),
         {'time': time.time(), 'gpu': gpu, 'checkpoint': str(cp), 'row': slot['fallback'],
          'dsrl_terminal': state, 'released_gpu': snapshot}, True)
    child = op.launch(slot['fallback_role'])
    identity = op.proc(child.pid)
    save(control / 'receipts' / ('gpu' + gpu + '-rlt-launched.json'),
         {'time': time.time(), 'identity': identity, 'row': slot['fallback']}, True)
    update_watch(p, gpu)
    state.update(state='RLT_STARTING', fallback_identity=identity, fallback_started_at=time.time())
    return child


def owner(control):
    p, b, op = checked(control)
    lock = (control / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    card_locks = []
    for gpu in ('6', '7'):
        card_lock = (control.parent / ('dsrl-rlt-gpu' + gpu + '-lease.lock')).open('a')
        fcntl.flock(card_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        card_locks.append(card_lock)
    assert not (control / 'owner-identity.json').exists(), 'Owner is single-use; no replay'
    release = read(p['release_receipt'])
    final_capture = read(p['stop_attempt'])['capture']
    assert release['all_released'] is True and release['owner'] == p['original_owner']
    assert set(release['targets']) == {'6', '7'}
    for gpu in ('6', '7'):
        assert p['slots'][gpu]['original'] == final_capture['targets'][gpu]
        assert release['targets'][gpu]['released'] is True
        assert not b.same(p['slots'][gpu]['original']['driver'])
        assert gpu_released(p, b, op, gpu, p['slots'][gpu]['original']['namespace'])[0]
    original_protected(p, b)
    me = b.proc(os.getpid())
    save(control / 'owner-identity.json', me, True)
    state = {'time': time.time(), 'owner': me, 'slots': {g: {'state': 'DEV_HOLD'} for g in ('6', '7')}}
    children, fallback = {}, {}
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        checked(control)
        original_protected(p, b)
        for gpu in ('6', '7'):
            row = state['slots'][gpu]
            try:
                if row['state'] == 'NEEDS_ATTENTION':
                    continue
                if row['state'] == 'DEV_HOLD':
                    pending = []
                    for path in sorted((control / 'requests').glob('*.json')):
                        req = read(path)
                        claimed = control / 'receipts' / (req['request_id'] + '-claimed.json')
                        if str(req['gpu']) == gpu and not claimed.exists():
                            pending.append((path, req, claimed))
                    assert len(pending) <= 1
                    if pending:
                        path, req, claimed = pending[0]
                        runtime = request_checked(p, req)
                        assert gpu_released(p, b, op, gpu, req.get('namespace'))[0]
                        save(claimed, {'time': time.time(), 'owner': me, 'request': str(path), 'sha256': sha(path)}, True)
                        env = environment(p, int(gpu), probe=req['kind'] == 'probe')
                        argv = [p['dsrl_python'], '-u', '-B', str(Path(p['code']) / 'dsrl_driver.py'),
                                '--control', str(control), '--request', str(path)]
                        with (runtime / 'driver.log').open('x') as log:
                            proc = subprocess.Popen(argv, cwd=p['dsrl_repo'], env=env, stdin=subprocess.DEVNULL,
                                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        identity = b.proc(proc.pid)
                        children[gpu] = {'proc': proc, 'identity': identity, 'request': req, 'identities': {str(proc.pid): identity}}
                        row.update(state='FORMAL' if req['kind'] == 'formal' else 'SMOKE', request_id=req['request_id'],
                                   driver=identity, namespace=req.get('namespace'), runtime=str(runtime), started_at=time.time())
                        save(control / 'receipts' / (req['request_id'] + '-launched.json'),
                             {'time': time.time(), 'identity': identity, 'argv': argv}, True)
                if gpu in children:
                    child = children[gpu]
                    actors, tree, actual, outside = tree_snapshot(p, b, op, child, child['request'])
                    if outside:
                        # Exact wrapper identity only; its finally block cleans its own Ray job.
                        if b.same(child['identity']):
                            fd = b.pidfd_open(child['identity']['pid'])
                            try:
                                assert b.same(child['identity'])
                                b.pidfd_signal(fd, signal.SIGTERM)
                            finally:
                                os.close(fd)
                        raise RuntimeError('Task process appeared on another GPU; exact wrapper stopped')
                    rc = child['proc'].poll()
                    if rc is not None:
                        released, snap, actors = gpu_released(p, b, op, gpu, row.get('namespace'))
                        live = [identity for identity in child['identities'].values() if b.same(identity)]
                        row.update(exit_code=rc, exited_at=row.get('exited_at', time.time()), state='RELEASE_CHECK')
                        if released and not live:
                            save(control / 'receipts' / (row['request_id'] + '-finished.json'),
                                 {'time': time.time(), 'exit_code': rc, 'gpu_released': snap,
                                  'namespace_clear': True, 'recorded_processes_clear': True}, True)
                            kind = child['request']['kind']
                            del children[gpu]
                            if kind == 'formal':
                                fallback[gpu] = restore(p, b, op, control, gpu, row)
                            else:
                                row['state'] = 'DEV_HOLD'
                if row['state'] == 'RLT_STARTING':
                    old = p['slots'][gpu]
                    op.ST = Path(p['fallback_plan']).parent
                    active = op.active_actors(op.checked(), old['fallback']['namespace'])
                    proof = b.first_round(old['fallback'], row['fallback_started_at'])
                    proc = fallback[gpu]
                    if proc.poll() is not None:
                        op.finished(old['fallback_role'], proc.returncode)
                        raise RuntimeError('Fallback RLT exited before first new round: ' + str(proc.returncode))
                    if proof and active and b.same(row['fallback_identity']):
                        op.validate_scoped_actors(op.checked(), active)
                        roots = {row['fallback_identity']['pid']} | {a['pid'] for a in active if a.get('pid')}
                        tree = op.process_tree(roots, p['uid'])
                        snapshot = b.gpu_snapshot()
                        assert not any(g['gpu'] != int(gpu) and any(x['pid'] in tree for x in g['processes'])
                                       for g in snapshot), 'Fallback RLT used another GPU'
                        save(control / 'receipts' / ('gpu' + gpu + '-rlt-restored.json'),
                             {'time': time.time(), 'identity': row['fallback_identity'], 'metrics': proof,
                              'run': old['fallback']['run'], 'namespace': old['fallback']['namespace']}, True)
                        row['state'] = 'RLT_RESTORED'
                if row['state'] == 'RLT_RESTORED':
                    proc = fallback[gpu]
                    if proc.poll() is not None:
                        op.ST = Path(p['fallback_plan']).parent
                        op.finished(p['slots'][gpu]['fallback_role'], proc.returncode)
                        save(control / 'receipts' / ('gpu' + gpu + '-rlt-finished.json'),
                             {'time': time.time(), 'exit_code': proc.returncode,
                              'identity': row['fallback_identity']}, True)
                        row['state'] = 'RLT_COMPLETE' if proc.returncode == 0 else 'NEEDS_ATTENTION'
                        if proc.returncode:
                            row['error'] = 'Restored RLT exited: ' + str(proc.returncode)
            except Exception:
                row.update(state='NEEDS_ATTENTION', error=traceback.format_exc())
        state['time'] = time.time()
        save(control / 'status.json', state)
        if all(row['state'] in ('RLT_RESTORED', 'RLT_COMPLETE') for row in state['slots'].values()) and not (control / 'lease-returned.json').exists():
            save(control / 'lease-returned.json', state, True)
        if all(row['state'] in ('RLT_COMPLETE', 'NEEDS_ATTENTION') for row in state['slots'].values()):
            save(control / 'owner-final.json', state, True)
            return
        time.sleep(3)
    state.update(time=time.time(), stopped=True,
                 stop_rule='Retain children; no signal or automatic fallback from a CPU owner stop')
    save(control / 'status.json', state)
    save(control / 'owner-stopped.json', state, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('owner')
    command = sub.add_parser('submit')
    command.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'owner':
        owner(args.control)
    else:
        submit(args.control, args.request)


if __name__ == '__main__':
    main()
