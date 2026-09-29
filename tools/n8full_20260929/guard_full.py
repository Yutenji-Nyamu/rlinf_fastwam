"""Inspect or update only this N8-full rollout's existing guard routing.

Execute remotely as chenyiteng. MODE defaults to inspect; allowlist/watch
mutations require EXPECTED={host: SHA256} from a fresh inspection. This changes
neither the root guard service nor shared Ray lifecycle.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat


ROOT = Path('/data/chenyiteng')
DEP = ROOT / 'deployment-20260929'
ST = DEP / 'n8full-watch'
ALLOWLIST = ROOT / 'security/ray-guard/training_allowlist.json'
WATCH = ROOT / 'deployment-20260927/rlt-six-task-watch/plan.json'
HOSTS = {'admin': ('sz1', 1003), 'h100-gpu01': ('sz3', 20001)}
TASKS = {'sz1': {'place_phone_stand', 'pick_dual_bottles'},
         'sz3': {'move_pillbottle_pad', 'rotate_qrcode'}}


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def owned(path, uid):
    path = Path(path)
    assert path.is_absolute() and path.is_file() and not path.is_symlink(), str(path)
    assert path.stat().st_uid == uid and path.resolve().is_relative_to(ROOT.resolve()), str(path)
    return path


def read(path, uid):
    return json.loads(owned(path, uid).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity_key(identity):
    return (int(identity['pid']), int(identity['uid']),
            int(identity.get('start', identity.get('start_ticks'))))


def alive(identity):
    pid, uid, start = identity_key(identity)
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] not in ('Z', 'X') and path.stat().st_uid == uid and int(fields[19]) == start
    except (FileNotFoundError, ProcessLookupError):
        return False


def load_plans(host, uid):
    index = DEP / f'n8full-{host}-prepared.json'
    ready = read(index, uid)
    assert ready['host'] == host and len(ready['plans']) == len(set(ready['plans'])) == 2
    result, names = [], set()
    for raw in ready['plans']:
        path = owned(raw, uid)
        assert path.name == 'plan.json' and path.parent.parent == DEP
        plan = read(path, uid)
        assert plan['uid'] == uid and plan['task'] in TASKS[host]
        assert set(plan['runs']) == {'stage1-full', 'clean', 'combo'}
        assert len(plan['old_runs']) == 2
        assert {row['source_key'] for row in plan['old_runs']} == {'clean', 'combo'}
        source = Path(plan['reuse_source_stage'])
        assert source.parent == ROOT / 'deployment-20260928'
        old_plan = read(source / 'plan.json', uid)
        assert old_plan['task'] == plan['task'] and old_plan['uid'] == uid
        assert old_plan['runs']['stage1-full'] == plan['runs']['stage1-full']
        assert old_plan['stage1_full_weights'] == plan['stage1_full_weights']
        for old in plan['old_runs']:
            assert Path(old['source_plan']) == source / 'plan.json'
            original = old_plan['runs'][old['source_key']]
            assert all(old[field] == original[field] for field in ('run', 'namespace', 'gpus'))
        for key in ('clean', 'combo'):
            row = plan['runs'][key]
            assert row['namespace'] == f"fx8full-{host}-{plan['task']}-{key}-0929"
            assert row['gpus'] == old_plan['runs'][key]['gpus'] and len(row['gpus']) == 1
            assert row['run'] != old_plan['runs'][key]['run']
            names.add(row['namespace'])
        management = plan['management_namespace']
        assert management == f"fx8full-{host}-{plan['task']}-cutover-0929"
        names.add(management)
        result.append((path, plan))
    assert {plan['task'] for _, plan in result} == TASKS[host]
    assert len(names) == 6
    assert len({old['run'] for _, plan in result for old in plan['old_runs']}) == 4
    assert len({row['run'] for _, plan in result for key, row in plan['runs'].items()
                if key in ('clean', 'combo')}) == 4
    return index, result, names


def replacements(plans, uid):
    targets, pending = {}, []
    for path, plan in plans:
        required = ('stage1-complete.json', 'old-stopped.json', 'formal-dispatched.json')
        if not all((path.parent / name).is_file() for name in required):
            pending.append(plan['task'])
            continue
        reuse = read(path.parent / 'stage1-complete.json', uid)
        weights = owned(plan['stage1_full_weights'], uid)
        assert reuse['reused_from'] == plan['reuse_source_stage']
        assert reuse['weights'] == str(weights) and reuse['bytes'] == weights.stat().st_size
        assert weights.stat().st_size > 1024 ** 3
        stopped = read(path.parent / 'old-stopped.json', uid)
        assert set(stopped['runs']) == {row['run'] for row in plan['old_runs']}
        dispatch = read(path.parent / 'formal-dispatched.json', uid)['drivers']
        assert set(dispatch) == {'clean', 'combo'}
        for key in ('clean', 'combo'):
            row = plan['runs'][key]
            identity = read(Path(row['run']) / 'runtime/driver-identity.json', uid)
            assert identity['namespace'] == row['namespace'] and identity['uid'] == uid
            assert identity_key(identity) == identity_key(dispatch[key]) and alive(identity)
            old = next(item for item in plan['old_runs'] if item['source_key'] == key)
            prior = read(Path(old['run']) / 'runtime/driver-identity.json', uid)
            assert identity_key(prior) == identity_key(old['identity'])
            assert prior['uid'] == uid and prior['namespace'] == old['namespace'] and not alive(prior)
            targets[old['run']] = (f"n8full-{plan['task']}__{key}",
                                  {field: row[field] for field in ('run', 'namespace', 'gpus')})
    return targets, pending


def replace_watch(before, targets):
    after = dict(before)
    after['runs'] = dict(before['runs'])
    switched = []
    for old_run, (new_key, new_row) in targets.items():
        keys = [key for key, row in before['runs'].items() if row['run'] == old_run]
        if not keys:
            assert after['runs'].get(new_key) == new_row, 'Old watch target missing: ' + old_run
            continue
        assert len(keys) == 1 and new_key not in after['runs'], 'Ambiguous watch replacement'
        del after['runs'][keys[0]]
        after['runs'][new_key] = new_row
        switched.append({'old_key': keys[0], 'old_run': old_run, 'new_key': new_key, **new_row})
    assert len(after['runs']) == len(before['runs'])
    assert all(after['runs'][key] == row for key, row in before['runs'].items()
               if row['run'] not in targets)
    return after, switched


def inspect(host, uid, plans, names):
    allow = owned(ALLOWLIST, uid).read_bytes()
    watch = owned(WATCH, uid).read_bytes()
    targets, pending = replacements(plans, uid)
    _, switched = replace_watch(json.loads(watch), targets)
    return {'host': host, 'mode': 'inspect', 'allowlist_sha256': digest(allow),
            'watch_sha256': digest(watch), 'namespaces_to_add': sorted(names - set(json.loads(allow)['namespaces'])),
            'ready_watch_replacements': switched, 'pending_tasks': pending,
            'root_script_changed': False, 'firewall_changed': False}


def mutate(host, uid, mode, expected, index, plans, names):
    assert isinstance(expected, str) and len(expected) == 64
    ST.mkdir(parents=True, exist_ok=True)
    assert not ST.is_symlink() and ST.stat().st_uid == uid and ST.resolve().is_relative_to(ROOT.resolve())
    with (ST / 'route.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = ALLOWLIST if mode == 'allowlist' else WATCH
        original = owned(target, uid).read_bytes()
        assert digest(original) == expected, 'Target changed since inspection'
        before = json.loads(original)
        contract = {'index_sha256': digest(index.read_bytes()),
                    'plan_sha256': {str(path): digest(path.read_bytes()) for path, _ in plans}}
        if mode == 'allowlist':
            after = dict(before)
            after['namespaces'] = sorted(set(before['namespaces']) | names)
            details = {'requested': sorted(names), 'added': sorted(names - set(before['namespaces']))}
            changed = bool(details['added'])
        else:
            targets, pending = replacements(plans, uid)
            assert not pending and len(targets) == 4, 'Wait for both exact formal pairs before routing watch'
            after, switched = replace_watch(before, targets)
            after['source_plans'] = sorted(set(before.get('source_plans', [])) | {str(path) for path, _ in plans})
            details = {'switched': switched, 'tracked_runs': list(after['runs'])}
            changed = bool(switched) or after != before
        directory = ST / mode
        if directory.exists():
            completed = read(directory / 'completed.json', uid)
            assert all(completed[key] == value for key, value in contract.items()), 'Different completed operation'
            assert completed['after_sha256'] == digest(original) and not changed
            return {'already_applied': True, **completed}
        if not changed:
            return {'host': host, 'mode': mode, 'unchanged': True, 'sha256': expected, **details}
        directory.mkdir()
        backup = directory / 'before.json'
        with backup.open('xb') as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        backup.chmod(stat.S_IMODE(target.stat().st_mode))
        save(directory / 'attempt.json', {'time': now(), 'host': host, 'mode': mode,
             'target': str(target), 'before_sha256': expected, **contract, **details})
        temporary = target.with_name(target.name + '.n8full-20260929-' + mode + '.tmp')
        save(temporary, after)
        temporary.chmod(stat.S_IMODE(target.stat().st_mode))
        assert digest(target.read_bytes()) == expected, 'Target changed before atomic replace'
        os.replace(temporary, target)
        assert read(target, uid) == after
        result = {'time': now(), 'host': host, 'mode': mode, 'target': str(target),
                  'before_sha256': expected, 'after_sha256': digest(target.read_bytes()),
                  'backup': str(backup), 'root_script_changed': False, 'firewall_changed': False,
                  'training_restarted': False, **contract, **details}
        save(directory / 'completed.json', result)
        return result


def main():
    host, uid = HOSTS[socket.gethostname()]
    assert os.getuid() == uid, 'Run as chenyiteng, not root'
    mode = globals().get('MODE', 'inspect')
    assert mode in ('inspect', 'allowlist', 'watch')
    index, plans, names = load_plans(host, uid)
    if mode == 'inspect':
        result = inspect(host, uid, plans, names)
    else:
        expected = globals().get('EXPECTED', {})
        assert isinstance(expected, dict) and host in expected
        result = mutate(host, uid, mode, expected[host], index, plans, names)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
