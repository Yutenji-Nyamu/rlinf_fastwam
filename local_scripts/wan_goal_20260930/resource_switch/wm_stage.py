"""Run one owned WM command, then require dedicated cleanup before Dojo resumes."""
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid
from common import UID, alive, atomic, identity, own_path, pidfd_open, pidfd_send, read, sha

TAG = 'WM_OWNER_TOKEN'


def validate_spec(spec):
    assert spec['schema'] == 1
    assert spec['physical_gpus'] == [4, 5, 6, 7]
    assert spec['max_seconds'] is None or (type(spec['max_seconds']) in (int, float) and spec['max_seconds'] > 0)
    assert 1 <= spec['cleanup_timeout_seconds'] <= 1800
    for key in ('command', 'cleanup_command'):
        cmd = spec[key]
        assert isinstance(cmd, list) and cmd and all(isinstance(x, str) and x for x in cmd)
        assert Path(cmd[0]).is_absolute() and Path(cmd[0]).is_file()
    run = own_path(spec['run_dir'], exists=False)
    assert not run.exists(), 'WM run directory must be new'
    cwd = own_path(spec['cwd']); assert cwd.is_dir()
    env = spec.get('environment', {})
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
    reserved = {TAG, 'CUDA_VISIBLE_DEVICES', 'RAY_ADDRESS', 'WAN_GOAL_RUN_DIR',
                'WAN_GOAL_CLEANUP_RECEIPT', 'WAN_GOAL_IDENTITIES', 'WAN_GOAL_CYCLE_ID',
                'WAN_GOAL_RAY_NAMESPACE', 'WAN_GOAL_RAY_TMPDIR', 'CLUSTER_NAMESPACE'}
    assert not reserved.intersection(env), 'Reserved ownership environment override'
    ray = spec['ray']
    assert ray['isolation'] == 'dedicated' and ray['address'] not in ('auto', '')
    address = re.fullmatch(r'127\.0\.0\.1:(\d+)', ray['address'])
    assert address and 1024 < int(address.group(1)) < 65536, 'Explicit private loopback Ray address required'
    assert isinstance(ray['namespace'], str) and ray['namespace']
    own_path(ray['temp_dir'], exists=False)
    assert not Path(ray['temp_dir']).exists()
    return spec


class Catalog:
    def __init__(self, run, token):
        self.run = run; self.token = token; self.rows = {}

    def add(self, row, origin):
        key = (row['pid'], row['start'], row['boot'])
        old = self.rows.get(key, {})
        row['origins'] = sorted(set(old.get('origins', [])) | {origin})
        row['observed_command_sha256'] = sorted(set(old.get('observed_command_sha256', [])) | {row['command_sha256']})
        self.rows[key] = row

    def scan(self):
        observed = {}
        unreadable_unclaimed = []
        registered_pids = {r['pid'] for r in self.rows.values()}
        needle = (TAG + '=' + self.token).encode()
        descendant_prefix = needle + b'.'
        for p in Path('/proc').iterdir():
            if not p.name.isdigit(): continue
            try:
                if p.stat().st_uid != UID: continue
                row = identity(int(p.name))
                if row['state'] in ('Z', 'X'): continue
                entries = (p / 'environ').read_bytes().split(b'\0')
                tagged = any(value == needle or value.startswith(descendant_prefix) for value in entries)
                observed[row['pid']] = row
                if tagged: self.add(row, 'owned-token-family')
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as error:
                if int(p.name) in registered_pids:
                    raise RuntimeError('Cannot verify registered WM PID ' + p.name +
                                       ' at ' + str(error.filename or p) + '; refusing to declare release') from error
                unreadable_unclaimed.append(dict(pid=int(p.name), path=str(error.filename or p),
                    reason='unregistered and unreadable; not claimed or signaled'))
                continue
        # A live exact registered parent also establishes descendants, even if
        # a launcher intentionally sanitizes its child environment.
        owned = {r['pid'] for r in self.rows.values() if alive(r)}
        while True:
            children = {pid for pid, row in observed.items() if row['ppid'] in owned} - owned
            if not children: break
            for pid in children: self.add(observed[pid], 'registered-parent-tree')
            owned |= children
        atomic(self.run / 'managed-identities.json', dict(time=time.time(), owner_token=self.token,
                managed_processes=list(self.rows.values()), unreadable_unclaimed=unreadable_unclaimed))

    def signal(self, row, sig):
        if not alive(row): return
        try:
            current = identity(row['pid'])
            assert (current['start'], current['boot'], current['uid']) == (row['start'], row['boot'], row['uid'])
            fd = pidfd_open(row['pid'])
        except (FileNotFoundError, ProcessLookupError):
            return
        try:
            if not alive(current, command=True): return
            try: pidfd_send(fd, sig)
            except ProcessLookupError: return
        finally:
            os.close(fd)

    def live(self):
        self.scan()
        return [row for row in self.rows.values() if alive(row)]


def run_stage(spec, cycle, attempt, gpu_processes, state, stop_requested):
    """Return only after the dedicated callback and independent release checks pass.

    Callback contract is documented in README.md. It must stop only its dedicated
    Ray instance, produce wm-cleanup.json, and return zero. No broad cleanup here.
    """
    validate_spec(spec)
    run = own_path(spec['run_dir'], exists=False); run.mkdir(mode=0o700, parents=True)
    parent_token = os.environ.get(TAG)
    token = (parent_token + '.' if parent_token else '') + uuid.uuid4().hex
    catalog = Catalog(run, token)
    ray = spec['ray']
    receipt_path = run / 'wm-cleanup.json'
    env = os.environ.copy(); env.update(spec.get('environment', {}))
    env.update({TAG: token, 'CUDA_VISIBLE_DEVICES': '4,5,6,7', 'RAY_ADDRESS': ray['address'],
                'WAN_GOAL_RUN_DIR': str(run), 'WAN_GOAL_CYCLE_ID': cycle.name,
                'WAN_GOAL_CLEANUP_RECEIPT': str(receipt_path),
                'WAN_GOAL_IDENTITIES': str(run / 'managed-identities.json'),
                'WAN_GOAL_RAY_NAMESPACE': ray['namespace'], 'CLUSTER_NAMESPACE': ray['namespace'],
                'WAN_GOAL_RAY_TMPDIR': ray['temp_dir']})
    atomic(run / 'owner.json', dict(time=time.time(), owner_token=token, cycle_id=cycle.name,
           parent_owner_token=parent_token,
           continuation=str(attempt), physical_gpus=[4, 5, 6, 7], ray=ray,
           parent_identity=identity(os.getpid()), command=spec['command']))
    state('RUNNING_WM', wm_run=str(run), owner_token=token)
    exit_code = None; outcome = 'failed'; error = None; child = None
    try:
        with (run / 'command.log').open('ab') as output:
            child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=env, stdout=output,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            try: catalog.add(identity(child.pid), 'wm-command-root')
            except (FileNotFoundError, ProcessLookupError): pass
            started = time.monotonic(); stopped_at = None
            while child.poll() is None:
                catalog.scan()
                expired = spec['max_seconds'] is not None and time.monotonic() - started >= spec['max_seconds']
                if stopped_at is None and (expired or stop_requested()):
                    outcome = 'timed_out' if expired else 'cancelled_to_resume_dojo'
                    stopped_at = time.monotonic()
                    for row in list(catalog.rows.values()):
                        if row['pid'] == child.pid: catalog.signal(row, signal.SIGTERM)
                if stopped_at is not None and time.monotonic() - stopped_at >= 30:
                    for row in list(catalog.rows.values()):
                        if row['pid'] == child.pid: catalog.signal(row, signal.SIGKILL)
                time.sleep(1)
            exit_code = child.returncode
            if stopped_at is None: outcome = 'completed' if exit_code == 0 else 'failed'
    except Exception as exc:
        error = repr(exc)
    finally:
        # Also runs if Popen or monitoring failed. A live command root is included
        # in the callback catalog; no Dojo launch until its identity is gone.
        catalog.scan()
        atomic(run / 'wm-exit.json', dict(time=time.time(), exit_code=exit_code, outcome=outcome, error=error))
        state('CLEANING_WM', wm_run=str(run), outcome=outcome)
        assert not receipt_path.exists(), 'Stale cleanup receipt'
        cleanup_started = time.time()
        with (run / 'cleanup.log').open('ab') as output:
            cleanup = subprocess.Popen(spec['cleanup_command'], cwd=spec['cwd'], env=env, stdout=output,
                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            cleanup_identity = None
            try:
                cleanup_identity = identity(cleanup.pid)
                catalog.add(cleanup_identity, 'cleanup-callback-root')
            except (FileNotFoundError, ProcessLookupError):
                pass
            deadline = time.monotonic() + spec['cleanup_timeout_seconds']
            while cleanup.poll() is None and time.monotonic() < deadline:
                catalog.scan(); time.sleep(1)
            if cleanup.poll() is None:
                assert cleanup_identity, 'Missing live cleanup callback identity'
                catalog.signal(cleanup_identity, signal.SIGTERM)
                try: cleanup.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    catalog.signal(cleanup_identity, signal.SIGKILL); cleanup.wait(timeout=10)
                raise RuntimeError('WM cleanup callback timed out; manual reconciliation required')
            assert cleanup.returncode == 0, 'WM cleanup callback failed'
        receipt = read(own_path(receipt_path))
        assert receipt['owner_token'] == token and receipt['cycle_id'] == cycle.name
        assert receipt['time'] >= cleanup_started and receipt['physical_gpus'] == [4, 5, 6, 7]
        assert receipt['processes_clear'] is True and receipt['gpus_released'] is True
        assert receipt['ray']['address'] == ray['address'] and receipt['ray']['namespace'] == ray['namespace']
        assert receipt['ray']['temp_dir'] == ray['temp_dir'] and receipt['ray']['stopped'] is True
        assert not catalog.live(), 'A registered WM process is still alive'
        assert not gpu_processes([4, 5, 6, 7]), 'Physical GPU 4–7 are not released'
        released = dict(time=time.time(), cycle_id=cycle.name, owner_token=token, outcome=outcome,
                        wm_exit_code=exit_code, error=error, all_workers_stopped=True,
                        managed_processes=list(catalog.rows.values()), cleanup_receipt=str(receipt_path),
                        cleanup_receipt_sha256=sha(receipt_path), physical_gpus=[4, 5, 6, 7],
                        processes_clear=True, gpus_released=True, next='RESUME_EXISTING_DOJO')
        atomic(attempt / 'wm-release.json', released)
    return released
