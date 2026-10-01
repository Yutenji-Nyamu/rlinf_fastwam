"""SZ2 EXPO lease owner; return the frozen four RLT runs on every exit.

Run as chenyiteng on h100-gpu02. The root operator updates work-heartbeat and
starts borrowed smoke-vN owners only after rlt-guardian.json reports PAUSED.
No shared Ray lifecycle changes and no process-group or process-name signals.
"""
import argparse
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback
from expo_process import pidfd_open, pidfd_send, pidfd_probe

ROOT = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001')
CYCLE = ROOT / 'rlt-cycle'
RLT_PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
UID = 20001
TTL = 15 * 60
RESTORE_WAIT = 30 * 60
INNER_STOP_WAIT = 120


class GuardianTerminated(RuntimeError):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def checked(path):
    path = Path(path).absolute()
    # /data/chenyiteng is an approved account alias on SZ2. Preserve that
    # caller-visible route while checking its resolved account containment.
    if not path.is_relative_to(ROOT) or not path.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError('Receipt path escapes the fixed EXPO root: ' + str(path))
    for item in (path, *path.parents):
        if item == ROOT.parent:
            break
        if item.exists() and (item.is_symlink() or item.stat().st_uid != UID):
            raise RuntimeError('Receipt path ownership/symlink differs: ' + str(item))
    return path


def read(path):
    return json.loads(checked(path).read_text())


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def identity(pid):
    path = Path('/proc') / str(int(pid))
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), uid=path.stat().st_uid, boot_id=boot(),
                start_ticks=int(fields[19]), state=fields[0])


def anchor(row, earliest=0):
    required = ('pid', 'uid', 'boot_id', 'start_ticks')
    if not isinstance(row, dict) or any(key not in row for key in required):
        raise RuntimeError('Incomplete managed process identity')
    out = {key: row[key] for key in required}
    if (out['uid'] != UID or out['boot_id'] != boot() or
            type(out['pid']) is not int or out['pid'] <= 1 or
            type(out['start_ticks']) is not int or out['start_ticks'] < earliest):
        raise RuntimeError('Managed process boot/UID/start boundary differs')
    return out


def alive(row):
    try:
        current = identity(row['pid'])
        return all(current[key] == row[key] for key in ('pid', 'uid', 'boot_id', 'start_ticks')) and current['state'] not in ('Z', 'X')
    except (FileNotFoundError, ProcessLookupError):
        return False


def pidfd_signal(row, sig):
    if not alive(row):
        return False
    try:
        fd = pidfd_open(row['pid'])
    except ProcessLookupError:
        return False
    try:
        # Opening the fd and the second /proc read bind the signal to the same
        # live incarnation; PID/PGID reuse cannot redirect it.
        if not alive(row):
            return False
        pidfd_send(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


class Guardian:
    def __init__(self, previous):
        self.me = identity(os.getpid())
        self.previous = previous
        self.phase = 'STARTING'
        self.requested = None
        self.critical = False
        self.closing = False
        self.paused = False
        self.managed = {}
        self.owners = {}
        self.operations = {}
        self.error = None
        self.terminal = 'not_started'
        self.restore = None
        self.logs = ROOT / 'guardian-logs'
        checked(self.logs)
        self.logs.mkdir(mode=0o700, exist_ok=True)
        spec = importlib.util.spec_from_file_location('expo_rlt_switch', checked(ROOT / 'rlt_switch.py'))
        self.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bridge)
        if self.bridge.KEYS != {'gpu4', 'gpu5', 'gpu6', 'gpu7'}:
            raise RuntimeError('Bridge GPU scope differs')

    def receipt(self, phase, **extra):
        self.phase = phase
        payload = dict(time=now(), state=phase, owner=self.me, cycle_id=CYCLE.name,
                       cycle_dir=str(CYCLE), rlt_paused=self.paused,
                       terminal_status=self.terminal, heartbeat=str(ROOT / 'work-heartbeat'),
                       heartbeat_ttl_seconds=TTL, operations=self.operations,
                       managed_processes=list(self.managed.values()),
                       inner_owners=list(self.owners.values()), error=self.error,
                       restoration=self.restore, requested_signal=self.requested)
        payload.update(extra)
        atomic(ROOT / 'rlt-guardian.json', payload)
        print(json.dumps(dict(time=payload['time'], state=phase,
                              terminal_status=self.terminal)), flush=True)

    def terminate(self, sig, _frame):
        self.requested = sig
        if not self.critical and not self.closing:
            raise GuardianTerminated('Guardian received signal ' + str(sig))

    def bridge_cli(self, action, timeout, release=None):
        mutation = action != 'status'
        if mutation and action in self.operations:
            raise RuntimeError('Refusing duplicate bridge mutation: ' + action)
        command = [RLT_PY, '-u', '-B', str(ROOT / 'rlt_switch.py'),
                   '--cycle-dir', str(CYCLE), '--previous-cycle', str(self.previous), action]
        if release is not None:
            command += ['--release-receipt', str(release)]
        label = action if mutation else 'status-' + str(time.time_ns())
        log_path = self.logs / (label + '.log')
        if mutation:
            self.operations[action] = dict(state='INTENT', time=now(), log=str(log_path))
            atomic(ROOT / ('rlt-guardian-' + action + '-intent.json'), self.operations[action])
            self.receipt(self.phase)
        self.critical = True
        child = None
        try:
            with log_path.open('xb') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, close_fds=True)
                child_anchor = anchor(identity(child.pid), self.me['start_ticks'])
                started = time.monotonic()
                while child.poll() is None:
                    if time.monotonic() - started > timeout:
                        pidfd_signal(child_anchor, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            pidfd_signal(child_anchor, signal.SIGKILL)
                            child.wait(timeout=10)
                        raise TimeoutError('Bridge ' + action + ' exceeded bound; inspect ' + str(log_path))
                    time.sleep(.5)
            with log_path.open('rb') as stream:
                stream.seek(max(0, log_path.stat().st_size - 2 * 1024 * 1024))
                lines = stream.read().decode(errors='replace').splitlines()
            result = next((json.loads(line) for line in reversed(lines)
                           if line.lstrip().startswith('{')), None)
            if child.returncode != 0 or not isinstance(result, dict):
                raise RuntimeError('Bridge ' + action + ' failed; inspect ' + str(log_path))
            if mutation:
                self.operations[action].update(state='FINISHED', exit_code=child.returncode, time=now())
            return result
        except BaseException as failure:
            if mutation:
                self.operations[action].update(state='FAILED_INSPECT_RECEIPTS', error=str(failure), time=now())
            raise
        finally:
            self.critical = False

    def collect(self):
        runs = ROOT / 'runs'
        if not runs.exists():
            return []
        completed = []
        for directory in sorted(runs.glob('smoke-v*')):
            if not re.fullmatch(r'smoke-v[0-9]+', directory.name):
                continue
            owner_path = directory / 'owner.json'
            if not owner_path.exists():
                continue
            meta = read(owner_path)
            raw = meta.get('owner', {})
            # Earlier unborrowed attempts and a prior boot are not this lease.
            if (not meta.get('rlt_borrowed') or raw.get('boot_id') != self.me['boot_id'] or
                    raw.get('start_ticks', -1) < self.me['start_ticks']):
                continue
            owner = anchor(raw, self.me['start_ticks'])
            scope = meta.get('scope')
            if not isinstance(scope, str) or not scope.startswith(directory.name + '-'):
                raise RuntimeError('Inner scope differs: ' + str(directory))
            key = (owner['pid'], owner['start_ticks'])
            previous = self.owners.get(key)
            if previous and (previous['scope'] != scope or previous['attempt'] != directory.name):
                raise RuntimeError('Inner owner registration changed')
            if alive(owner):
                argv = (Path('/proc') / str(owner['pid']) / 'cmdline').read_bytes().split(b'\0')
                words = [item.decode(errors='replace') for item in argv if item]
                expected_script = str(ROOT / 'source/tools/expo_smoke_owner.py')
                if (expected_script not in words or '--attempt' not in words or
                        words[words.index('--attempt') + 1] != directory.name or
                        '--rlt-paused' not in words or not alive(owner)):
                    raise RuntimeError('Live inner command identity differs: ' + str(directory))
            self.owners[key] = dict(owner, scope=scope, attempt=directory.name, receipt=str(owner_path))
            self.managed[key] = owner
            for phase in ('fresh', 'resume'):
                path = directory / phase / 'process-roster.json'
                if not path.exists():
                    continue
                roster = read(path)
                if roster.get('scope') != scope or roster.get('phase') != phase:
                    raise RuntimeError('Inner roster provenance differs: ' + str(path))
                root = anchor(roster['root'], owner['start_ticks'])
                if root['pid'] == owner['pid']:
                    raise RuntimeError('Inner driver equals owner')
                rows = roster.get('registered')
                if not isinstance(rows, list) or not rows:
                    raise RuntimeError('Inner roster is empty')
                root_seen = False
                for row in rows:
                    registered = anchor(row, root['start_ticks'])
                    if registered == root:
                        root_seen = True
                    self.managed[(registered['pid'], registered['start_ticks'])] = registered
                if not root_seen:
                    raise RuntimeError('Inner roster omits its exact driver')
            final_path = directory / 'final.json'
            if final_path.exists():
                final = read(final_path)
                if final.get('scope') != scope or anchor(final['owner']) != owner:
                    raise RuntimeError('Final receipt owner/scope differs')
                if final.get('smoke_passed') is True and final.get('rlt_borrowed') is True:
                    completed.append(str(final_path))
        return completed

    def wait_lease(self):
        paused_at = time.time()
        last_count = -1
        while True:
            completed = self.collect()
            if completed:
                self.terminal = 'completed'
                self.receipt('SMOKE_PASSED', completion_receipts=completed)
                return
            path = ROOT / 'work-heartbeat'
            updated = checked(path).stat().st_mtime if path.exists() else paused_at
            if updated > time.time() + 60:
                raise RuntimeError('Work heartbeat has a future timestamp')
            age = time.time() - updated
            if age > TTL:
                self.terminal = 'timed_out' if self.owners else 'not_started'
                self.receipt('LEASE_EXPIRED', heartbeat_age_seconds=age)
                return
            if last_count != len(self.managed):
                self.receipt('PAUSED', heartbeat_age_seconds=age)
                last_count = len(self.managed)
            time.sleep(2)

    def stop_inner(self):
        self.receipt('STOPPING_EXPO')
        deadline = time.monotonic() + INNER_STOP_WAIT
        sent = set()
        while True:
            self.collect()
            live_owners = {key: row for key, row in self.owners.items() if alive(row)}
            for key, row in live_owners.items():
                if key not in sent:
                    pidfd_signal(row, signal.SIGTERM)
                    sent.add(key)
            if not live_owners:
                # A final scan includes descendants registered during TERM.
                self.collect()
                live = [row for row in self.managed.values() if alive(row)]
                if live:
                    raise RuntimeError('Inner owner exited with exact live descendants; no direct descendant signals: ' + str(live))
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Inner owner did not finish its finally cleanup within 120 seconds')
            time.sleep(1)
        helper = self.bridge.frozen(CYCLE)
        occupied = helper.gpu_processes([4, 5, 6, 7])
        if occupied:
            raise RuntimeError('GPU4-7 still occupied; refusing restoration: ' + str(occupied))
        if not self.managed:
            self.terminal = 'not_started'
        elif self.terminal == 'not_started':
            self.terminal = 'failed'
        release = dict(time=now(), cycle_id=CYCLE.name, terminal_status=self.terminal,
                       all_workers_stopped=True,
                       managed_processes=[dict(pid=row['pid'], uid=row['uid'],
                                               start=row['start_ticks'], boot=row['boot_id'])
                                          for row in self.managed.values()],
                       cleanup_receipt=dict(inner_owners_stopped=True,
                                            exact_registered_descendants_stopped=True,
                                            gpus_released=[4, 5, 6, 7], signals='pidfd SIGTERM to inner owners only'))
        path = ROOT / 'rlt-release.json'
        atomic(path, release)
        verified = helper.validate_release(CYCLE, path)
        atomic(ROOT / 'rlt-release-verified.json', verified)
        return path

    def restore_rlt(self):
        release = self.stop_inner()
        self.receipt('RESTORING')
        result = self.bridge_cli('resume', 600, release)
        self.restore = dict(dispatched=result, first_round_verified=False)
        self.receipt('WAITING_FIRST_ROUND')
        deadline = time.monotonic() + RESTORE_WAIT
        last_status_error = None
        while time.monotonic() < deadline:
            try:
                status = self.bridge_cli('status', min(90, max(1, deadline - time.monotonic())))
                atomic(ROOT / 'rlt-guardian-latest-status.json', status)
                if status.get('all_first_rounds_verified') is True:
                    self.restore.update(first_round_verified=True,
                                        proof=str(CYCLE / 'rlt-first-round.json'), time=now())
                    self.receipt('RESTORED')
                    return
                last_status_error = None
            except Exception as failure:
                last_status_error = str(failure)
            time.sleep(min(20, max(0, deadline - time.monotonic())))
        raise TimeoutError('RLT dispatched but four first rounds not verified within 30min; inspect latest status. Last status error: ' + str(last_status_error))

    def run(self):
        signal.signal(signal.SIGTERM, self.terminate)
        signal.signal(signal.SIGINT, self.terminate)
        try:
            # The root operator may finish the CPU-only preparation first.
            # Adopt only its frozen, complete, not-yet-stopped plan; never replay
            # a prior stop/resume or silently change that prepared lineage.
            for action in ('prepare', 'stop', 'resume'):
                if (ROOT / ('rlt-guardian-' + action + '-intent.json')).exists():
                    raise RuntimeError('Prior guardian intent exists; inspect before a new guardian')
            self.receipt('PREPARING')
            if (CYCLE / 'plan.json').exists():
                for name in ('rlt-stopped.json', 'resumed-dispatched.json', 'dojo-release-verified.json'):
                    if (CYCLE / name).exists():
                        raise RuntimeError('Prepared cycle already stopped/resumed; inspect exact prior owner')
                self.bridge.frozen(CYCLE)
                prepared = read(CYCLE / 'prepared.json')
                self.operations['prepare'] = dict(state='ADOPTED_CPU_PREPARE', time=now(),
                                                   receipt=str(CYCLE / 'prepared.json'))
            else:
                if CYCLE.exists() and any(p.name != 'operation.lock' for p in CYCLE.iterdir()):
                    raise RuntimeError('CPU preparation is partial; no RLT stop will be attempted')
                prepared = self.bridge_cli('prepare', 600)
            if not prepared.get('prepared') or prepared.get('gpu_used_for_check') is not False:
                raise RuntimeError('CPU prepare receipt incomplete')
            if prepared.get('cycle_id') != CYCLE.name or prepared.get('run_count') != 4:
                raise RuntimeError('Prepared cycle/run count differs')
            if self.requested:
                raise GuardianTerminated('Termination requested during CPU preparation')
            self.receipt('STOPPING_RLT')
            self.bridge_cli('stop', 900)
            stopped = read(CYCLE / 'rlt-stopped.json')
            frozen = read(CYCLE / 'bridge-frozen.json')
            if (frozen['boot'] != self.me['boot_id'] or not stopped.get('all_original_drivers_stopped') or
                    not stopped.get('all_original_namespaces_empty') or set(stopped.get('gpus_released', [])) != {4, 5, 6, 7}):
                raise RuntimeError('RLT stop receipt incomplete')
            self.paused = True
            self.receipt('PAUSED')
            if self.requested:
                raise GuardianTerminated('Termination requested during RLT stop')
            self.wait_lease()
        except BaseException as failure:
            self.error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
            self.terminal = 'failed' if self.owners else 'not_started'
        finally:
            self.closing = True
            try:
                # A stop helper may return nonzero after writing its complete
                # stop receipt. Only the validated frozen cycle can restore it.
                if not self.paused and 'stop' in self.operations and (CYCLE / 'rlt-stopped.json').exists():
                    helper = self.bridge.frozen(CYCLE)
                    stopped = read(CYCLE / 'rlt-stopped.json')
                    self.paused = bool(stopped.get('all_original_drivers_stopped') and
                                       stopped.get('all_original_namespaces_empty') and
                                       set(stopped.get('gpus_released', [])) == {4, 5, 6, 7})
                    if not self.paused:
                        raise RuntimeError('Partial stop receipt; manual frozen-cycle recovery required')
                if self.paused:
                    self.restore_rlt()
                elif 'stop' in self.operations:
                    raise RuntimeError('Stop mutation has no complete receipt; inspect bridge stop log before frozen-cycle recovery')
                else:
                    self.receipt('FAILED_PREPARE_NO_GPU_TOUCH', action_required='Inspect prepare log/receipt; no RLT stop was attempted')
            except BaseException as failure:
                self.error = dict(type='RESTORATION_REQUIRES_ACTION', message=str(failure),
                                  previous_error=self.error, traceback=traceback.format_exc())
                self.receipt('FAILED_ACTION_REQUIRED', action_required='Inspect exact frozen cycle, guardian logs and inner cleanup receipts; do not repeat stop/resume blindly')
        return 0 if self.restore and self.restore.get('first_round_verified') and self.terminal == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-cycle', type=Path,
                        default=Path('/data/chenyiteng/projects/robodojo-openwam/runs/rlt-cycle-20260930-single-v4'))
    args = parser.parse_args()
    if os.getuid() != UID or socket.gethostname() != 'h100-gpu02':
        raise RuntimeError('Guardian requires the exact SZ2 chenyiteng account')
    checked(ROOT)
    fd = os.open(checked(ROOT / 'resource-owner.lock'), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            pidfd_probe()
            guardian = Guardian(args.previous_cycle)
        except BaseException as failure:
            atomic(ROOT / 'rlt-guardian.json', dict(time=now(), state='FAILED_PREPARE_NO_GPU_TOUCH',
                   error=dict(type=type(failure).__name__, message=str(failure)),
                   action_required='Fix CPU guardian/bridge preflight; no RLT stop was attempted'))
            raise
        raise SystemExit(guardian.run())


if __name__ == '__main__':
    main()
