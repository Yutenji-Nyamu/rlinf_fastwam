"""One-shot, exact-file checkpoint pruning authorized on 2026-09-06."""
import argparse, datetime, json, os, re, stat
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('manifest')
p.add_argument('--execute', action='store_true')
a = p.parse_args()
base = Path('/data/chenyiteng/results')
audit = base / ('server-maintenance-20260906/' + Path(a.manifest).stem + '-prune-audit.jsonl')
assert os.getuid() == 1003 and base.resolve() == base
plan = json.loads(Path(a.manifest).read_text())
assert len(plan['targets']) == plan['files'] == len({x['path'] for x in plan['targets']})

def free():
    return {str(q): os.statvfs(q).f_bavail * os.statvfs(q).f_frsize for q in (Path('/data'), Path('/home'))}

def live_references():
    opened, commands = set(), []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            commands.append((proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
            for fd in (proc / 'fd').iterdir():
                try:
                    opened.add(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            pass
    return opened, commands

def referenced(generation, commands):
    # Path-component boundary: step_10 must not match step_100.
    return any(generation + '/' in cmd or generation in cmd.split() for cmd in commands)

def unchanged(f):
    q = Path(f['path'])
    assert q.is_relative_to(base) and q != base and q.resolve() == q
    s = q.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid == 1003 and s.st_nlink == 1
    assert (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns) == (f['inode'], f['device'], f['bytes'], f['mtime_ns'])
    return s

protected = {f['path']: f for f in plan['protected_files']}
for f in protected.values():
    unchanged(f)
opened, commands = live_references()
selected, skipped = [], []
for f in plan['targets']:
    q = Path(f['path'])
    cr = Path(f['checkpoint_root'])
    assert cr.is_relative_to(base) and cr.name == 'checkpoints'
    generation = cr / ('global_step_' + str(f['step']))
    assert q.is_relative_to(generation) and q != generation and f['bytes'] > 1024**3
    assert f['path'] not in protected
    smoke = bool(re.search(r'(^|[/_-])smokes?([/_-]|\d|$)', str(cr), re.I))
    assert smoke == f['is_smoke']
    if not smoke:
        assert str(cr).startswith(('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/', '/data/chenyiteng/results/rlinf-shenzhen/online-bc/', '/data/chenyiteng/results/rlinf-current-dsrl/'))
        latest = max(int(v.name.removeprefix('global_step_')) for v in cr.glob('global_step_*') if v.is_dir())
        assert f['step'] < latest
    reason = None
    if referenced(str(generation), commands):
        reason = 'active process still references this generation'
    elif str(q) in opened:
        reason = 'open file'
    try:
        unchanged(f)
    except (OSError, AssertionError):
        reason = 'stat/ownership/path changed; not deleted'
    if reason:
        skipped.append({'path': str(q), 'reason': reason, 'bytes': f['bytes']})
    else:
        selected.append(f)

header = {'event': 'preflight', 'time': datetime.datetime.now().astimezone().isoformat(), 'execute': a.execute,
          'selected_files': len(selected), 'allocated_gib': sum(f['allocated'] for f in selected) / 1024**3,
          'skipped': skipped, 'protected_files': len(protected), 'free_before': free()}
print(json.dumps(header), flush=True)
if not a.execute:
    raise SystemExit(0)
assert not audit.exists(), 'Never replay an already-started destructive batch'
deleted = []
with audit.open('x') as log:
    def record(row):
        log.write(json.dumps(row) + '\n'); log.flush(); os.fsync(log.fileno())
    record(header)
    for f in selected:
        # Recheck per file immediately before unlink. No directory operations.
        opened, commands = live_references()
        generation = str(Path(f['checkpoint_root']) / ('global_step_' + str(f['step'])))
        if f['path'] in opened or referenced(generation, commands):
            row = {'event': 'skip_late_reference', 'path': f['path']}
            record(row); print(json.dumps(row), flush=True); continue
        unchanged(f)
        record({'event': 'unlink_intent', 'file': f})
        os.unlink(f['path'])
        assert not Path(f['path']).exists()
        deleted.append(f)
        row = {'event': 'deleted', 'path': f['path'], 'allocated': f['allocated']}
        record(row); print(json.dumps(row), flush=True)
    for f in protected.values():
        unchanged(f)
    result = {'event': 'complete', 'time': datetime.datetime.now().astimezone().isoformat(),
              'deleted_files': len(deleted), 'deleted_allocated_gib': sum(f['allocated'] for f in deleted) / 1024**3,
              'protected_files_verified': len(protected), 'skipped': skipped, 'free_after': free(),
              'recovery': 'Unlinked large files are not recoverable from Git/logs. All directories and smaller files remain.'}
    record(result); print(json.dumps(result), flush=True)
