"""Linux process receipts and narrowly scoped cleanup for one Dojo sweep.

No process-name matching, process-group killing, or other-user signals.
Children inherit both DOJO_SWEEP_ID and the exact ROBODOJO_RUN_ID.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import pwd
from pathlib import Path
import signal
import threading
import time
import traceback
import uuid


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def error_context(error):
    """Preserve actionable paths and traceback without reading process secrets."""
    return {"type": type(error).__name__, "repr": repr(error), "message": str(error),
            "errno": getattr(error, "errno", None),
            "filename": getattr(error, "filename", None),
            "filename2": getattr(error, "filename2", None),
            "pid": getattr(error, "process_guard_pid", None),
            "operation": getattr(error, "process_guard_operation", None),
            "diagnostic_path": getattr(error, "process_guard_diagnostic_path", None),
            "diagnostic_write_error": getattr(error, "process_guard_diagnostic_write_error", None),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}


class ProcessGuard:
    def __init__(self, sweep_id: str, uid: int, receipts: Path):
        if os.getuid() != uid:
            raise RuntimeError(f"UID mismatch: current={os.getuid()}, expected={uid}")
        self.sweep_id = sweep_id
        self.uid = uid
        self.receipts = receipts
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.known_pids = set()
        self.known_lock = threading.Lock()
        # Only proven exact cleanup targets enter this map. It allows bounded
        # fresh reads during exit, never signalling from stale identity.
        self.termination_pending = {}
        self.known_path = self.receipts / 'ownership-known.jsonl'
        anchor_path = self.receipts / 'ownership-anchor.json'
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        candidate = {'sweep_id':sweep_id, 'uid':uid, 'boot_id':self.boot_id,
                     'start_ticks':int(fields[19]), 'created_at':time.time()}
        # Separate from controller.lock: constructors can run before that lock.
        with (self.receipts / 'ownership.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if anchor_path.is_file():
                previous = json.loads(anchor_path.read_text())
                if previous['sweep_id'] != sweep_id or previous['uid'] != uid:
                    raise RuntimeError('Ownership anchor belongs to another sweep/UID')
                if previous['boot_id'] == self.boot_id:
                    candidate = previous
            else:
                # An old manifest without its anchor cannot prove which orphan
                # processes predate this sweep. Do not invent a later boundary.
                if (self.receipts.parent / 'plan.json').exists():
                    raise RuntimeError('Existing sweep is missing its process ownership anchor')
            atomic_json(anchor_path, candidate)
            self.anchor = candidate
            if self.known_path.is_file():
                for line in self.known_path.read_text().splitlines():
                    row = json.loads(line)
                    if row['sweep_id'] != sweep_id or row['uid'] != uid:
                        raise RuntimeError('Ownership history belongs to another sweep/UID')
                    if row['boot_id'] == self.boot_id:
                        self.known_pids.add((row['pid'], row['start_ticks']))

    def remember(self, pid: int, start_ticks: int):
        """Persist proven ownership before cleanup, including detached children."""
        with self.known_lock:
            if (pid, start_ticks) not in self.known_pids:
                row = {'pid':pid, 'start_ticks':start_ticks, 'uid':self.uid,
                       'sweep_id':self.sweep_id, 'boot_id':self.boot_id}
                with self.known_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                self.known_pids.add((pid, start_ticks))

    def predates_sweep(self, pid: int, start_ticks: int) -> bool:
        with self.known_lock:
            return start_ticks < self.anchor['start_ticks'] and (pid, start_ticks) not in self.known_pids

    def external_ssh_session(self, pid: int, start_ticks: int, parent_pid: int) -> bool:
        """Recognize a root-supervised login, never an owned worker by name."""
        with self.known_lock:
            if (pid, start_ticks) in self.known_pids:
                return False
        path, parent = Path('/proc') / str(pid), Path('/proc') / str(parent_pid)
        user = pwd.getpwuid(self.uid).pw_name
        try:
            if parent.stat().st_uid != 0:
                return False
            if (path / 'comm').read_text().strip() != 'sshd' or (parent / 'comm').read_text().strip() != 'sshd':
                return False
            child_command = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
            parent_command = (parent / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
            return (child_command.startswith(f'sshd: {user}@')
                    and parent_command == f'sshd: {user} [priv]')
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            return False

    def identity(self, pid: int):
        # Re-read through short execve transitions; persistent errors still stop.
        attempt = 0
        while True:
            try:
                return self._identity_once(pid)
            except PermissionError as error:
                attempt += 1
                if attempt >= 10 and not self.exit_read_retry_allowed(pid, error):
                    # Keep the same failure policy; record why identity was
                    # unavailable before worker-level handlers erase the path.
                    self.record_permission_error(pid, error)
                    raise
                time.sleep(0.05)

    def exit_read_retry_allowed(self, pid, error):
        """Wait for an exact cleanup target during exit; never assert ownership."""
        if getattr(error, 'filename', None) not in (
                f'/proc/{pid}', f'/proc/{pid}/stat', f'/proc/{pid}/environ', f'/proc/{pid}/cmdline'):
            return False
        now = time.monotonic()
        with self.known_lock:
            pending = [(start, deadline) for (target_pid, start), deadline in self.termination_pending.items()
                       if target_pid == pid and deadline > now]
        if not pending:
            return False
        try:
            fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
            start_ticks = int(fields[19])
        except (FileNotFoundError, ProcessLookupError):
            # A fresh _identity_once call will verify disappearance or PID reuse.
            return True
        except (PermissionError, ValueError, IndexError):
            return False
        return any(start == start_ticks for start, _deadline in pending)

    def record_permission_error(self, pid, error):
        """Diagnostic only. Unreadable metadata never proves ownership."""
        path = Path('/proc') / str(pid)
        record = {"time": time.time(), "boot_id": self.boot_id,
                  "sweep_id": self.sweep_id, "expected_uid": self.uid,
                  "pid": pid, "error": error_context(error), "samples": []}
        for _ in range(2):
            sample = {}
            try:
                sample['proc_owner_uid'] = path.stat().st_uid
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                sample.update(state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]))
                with self.known_lock:
                    sample['previously_proven_owned'] = (pid, sample['start_ticks']) in self.known_pids
                allowed = {'State', 'PPid', 'Uid', 'Gid', 'TracerPid', 'CoreDumping', 'NoNewPrivs'}
                sample['status'] = {name: value.strip() for line in (path / 'status').read_text().splitlines()
                                    for name, sep, value in [line.partition(':')] if sep and name in allowed}
            except (OSError, ValueError, IndexError) as diagnostic_error:
                sample['read_error'] = {"type": type(diagnostic_error).__name__,
                                        "errno": getattr(diagnostic_error, 'errno', None),
                                        "filename": getattr(diagnostic_error, 'filename', None)}
            record['samples'].append(sample)
        destination = self.receipts / ('permission-' + str(time.time_ns()) + '-' + uuid.uuid4().hex[:8] + '.json')
        try:
            atomic_json(destination, record)
            error.process_guard_diagnostic_path = str(destination)
        except OSError as diagnostic_error:
            # Never replace the original exception with a diagnostic-write error.
            error.process_guard_diagnostic_write_error = str(diagnostic_error)

    def _identity_once(self, pid: int):
        path = Path("/proc") / str(pid)
        operation = "proc_directory_stat"
        try:
            if path.stat().st_uid != self.uid:
                return None
            operation = "proc_stat"
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            start_ticks = int(fields[19])
            environ = {}
            try:
                operation = "proc_environ"
                environment_bytes = (path / "environ").read_bytes()
            except PermissionError:
                # Exiting processes can lose readable environ between the
                # initial stat and this read. Confirm death rather than treating
                # it as a live, unverifiable worker (or signalling a reused PID).
                for delay in (0, 0.02, 0.05):
                    if delay:
                        time.sleep(delay)
                    latest = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                    if latest[0] in ('Z', 'X') or int(latest[19]) != start_ticks:
                        return None
                # Same UID is not ownership. A demonstrably older, never-owned
                # process cannot be one of this sweep's descendants.
                if self.predates_sweep(pid, start_ticks):
                    return None
                # An SSH login's root monitor establishes that it belongs to
                # sshd, even when the login was opened after this sweep began.
                if self.external_ssh_session(pid, start_ticks, int(fields[1])):
                    return None
                raise
            for field in environment_bytes.split(b"\0"):
                name, separator, value = field.partition(b"=")
                if separator and name in (b"DOJO_SWEEP_ID", b"ROBODOJO_RUN_ID", b"DOJO_ROLE"):
                    environ[name.decode()] = value.decode(errors="replace")
            if environ.get("DOJO_SWEEP_ID") != self.sweep_id:
                return None
            operation = "ownership_history_append"
            self.remember(pid, start_ticks)
            operation = "proc_cmdline"
            command = (path / "cmdline").read_bytes()
            return {
                "pid": pid, "uid": self.uid, "ppid": int(fields[1]),
                "start_ticks": start_ticks,
                "command_sha256": hashlib.sha256(command).hexdigest(),
                "command": command.replace(b"\0", b" ").decode(errors="replace"),
                "sweep_id": self.sweep_id,
                "run_id": environ.get("ROBODOJO_RUN_ID"),
                "role": environ.get("DOJO_ROLE"),
            }
        except (FileNotFoundError, ProcessLookupError):
            return None
        except PermissionError as error:
            error.process_guard_pid = pid
            error.process_guard_operation = operation
            raise
        # Other unreadable processes from this sweep's lifetime still fail.

    def scan(self, run_id: str | None = None):
        found = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit() or int(path.name) == os.getpid():
                continue
            row = self.identity(int(path.name))
            if row and (run_id is None or row["run_id"] == run_id):
                found.append(row)
        return sorted(found, key=lambda row: (row["start_ticks"], row["pid"]))

    @staticmethod
    def same_process(previous, current):
        keys = ("pid", "uid", "start_ticks", "command_sha256", "sweep_id", "run_id")
        return current is not None and all(previous[key] == current[key] for key in keys)

    def send(self, target, sig):
        """Open pidfd where supported, then revalidate identity before signalling."""
        fd = None
        operation = 'pidfd_open'
        try:
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                try:
                    fd = os.pidfd_open(target["pid"])
                except OSError as error:
                    if error.errno not in (22, 38):  # old kernel: EINVAL / ENOSYS
                        raise
            operation = 'signal_identity_recheck'
            if not self.same_process(target, self.identity(target["pid"])):
                return None
            if fd is not None:
                operation = 'pidfd_send_signal'
                signal.pidfd_send_signal(fd, sig)
            else:
                operation = 'kill_exact_pid'
                os.kill(target["pid"], sig)
            with self.known_lock:
                # This deadline never slides when identity reads fail. Even an
                # owned process still raises if it stays unreadable for 8 seconds.
                self.termination_pending[(target['pid'], target['start_ticks'])] = time.monotonic() + 8
            return {**target, "signal": signal.Signals(sig).name, "pidfd": fd is not None,
                    "signalled_at": time.time()}
        except ProcessLookupError:
            return None
        except PermissionError as error:
            if not hasattr(error, 'process_guard_operation'):
                error.process_guard_pid = target['pid']
                error.process_guard_operation = operation
                self.record_permission_error(target['pid'], error)
            raise
        finally:
            if fd is not None:
                os.close(fd)

    def cleanup(self, run_id: str | None, reason: str, grace: float = 20):
        """Return only after this exact scope is empty; retain a unique receipt."""
        receipt_path = self.receipts / (f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.json")
        receipt = {"sweep_id": self.sweep_id, "run_id": run_id, "uid": self.uid,
                   "reason": reason, "started_at": time.time(),
                   "initial_targets": [], "initial_scan_complete": False,
                   "actions": [], "state": "STARTING"}
        # Even the initial scan can fail. Persist its exact scope before scanning.
        atomic_json(receipt_path, receipt)
        try:
            return self._cleanup_impl(run_id, grace, receipt_path, receipt)
        except Exception as error:
            receipt.update(state="ERROR", finished_at=time.time(), error=error_context(error))
            try:
                atomic_json(receipt_path, receipt)
            except OSError as diagnostic_error:
                error.process_guard_diagnostic_write_error = str(diagnostic_error)
            raise

    def _cleanup_impl(self, run_id, grace, receipt_path, receipt):
        receipt['initial_targets'] = self.scan(run_id)
        receipt['initial_scan_complete'] = True
        receipt['state'] = 'CLEANING'
        atomic_json(receipt_path, receipt)
        # A signalled parent may terminate children before we reach them. Retain
        # all targets proved by the initial scan for the same bounded exit wait;
        # their cached rows are never used instead of a fresh signal identity.
        with self.known_lock:
            exit_deadline = time.monotonic() + 8
            for target in receipt['initial_targets']:
                self.termination_pending[(target['pid'], target['start_ticks'])] = exit_deadline
        for sig, wait_seconds in ((signal.SIGTERM, grace), (signal.SIGKILL, 5)):
            deadline = time.monotonic() + wait_seconds
            sent = set()
            while True:
                targets = self.scan(run_id)
                # New descendants are discovered even if their parent exited.
                for target in reversed(targets):
                    key = (target["pid"], target["start_ticks"], target["command_sha256"])
                    if key in sent:
                        continue
                    action = self.send(target, sig)
                    sent.add(key)
                    if action:
                        receipt["actions"].append(action)
                        atomic_json(receipt_path, receipt)
                if not targets:
                    time.sleep(0.2)
                    if not self.scan(run_id):
                        break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.5)
            if not self.scan(run_id):
                break
        receipt["remaining"] = self.scan(run_id)
        receipt["finished_at"] = time.time()
        receipt['state'] = 'REMAINING' if receipt['remaining'] else 'EMPTY'
        atomic_json(receipt_path, receipt)
        if receipt["remaining"]:
            raise RuntimeError(f"Dojo processes remain; see {receipt_path}")
        return str(receipt_path)
