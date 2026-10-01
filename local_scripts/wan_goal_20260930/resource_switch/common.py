"""Small helpers for a new SZ3 owner; existing frozen runtime files stay untouched."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/chenyiteng')
UID = 20001
HELPER = 'rlt_cycle_sz3.py'


def _pidfd_syscall(number, *arguments):
    """Linux native x86_64 ABI only; never fall back to signalling a bare PID.

    Numbers 424/434: Linux v6.8 arch/x86/entry/syscalls/syscall_64.tbl.
    https://github.com/torvalds/linux/blob/v6.8/arch/x86/entry/syscalls/syscall_64.tbl#L348
    """
    import ctypes
    if (sys.platform != 'linux' or platform.machine() != 'x86_64' or
            ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8):
        raise RuntimeError('pidfd native binding unavailable; syscall fallback requires Linux x86_64 LP64')
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def pidfd_open(pid):
    native = getattr(os, 'pidfd_open', None)
    if callable(native):
        return native(int(pid), 0)
    import ctypes
    fd = _pidfd_syscall(434, ctypes.c_int(int(pid)), ctypes.c_uint(0))
    try:
        os.set_inheritable(fd, False)
    except Exception:
        os.close(fd)
        raise
    return fd


def pidfd_send(fd, sig):
    native = getattr(signal, 'pidfd_send_signal', None)
    if callable(native):
        return native(int(fd), int(sig), None, 0)
    import ctypes
    return _pidfd_syscall(424, ctypes.c_int(int(fd)), ctypes.c_int(int(sig)),
                          ctypes.c_void_p(None), ctypes.c_uint(0))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.chmod(0o600)
    os.replace(tmp, path)


def exclusive(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())


def own_path(path, exists=True):
    path = Path(path).absolute()
    assert path.resolve().is_relative_to(ROOT.resolve()), 'Path outside account data root'
    assert not path.is_symlink(), 'Symlink endpoint refused'
    if exists:
        assert path.exists() and path.stat().st_uid == UID, 'Missing/unowned path'
    # Resolve for containment only. /data may be an approved bind/symlink route;
    # keep caller-visible paths stable for launcher/config identity contracts.
    return path


def account():
    assert os.getuid() == UID and socket.gethostname() == 'h100-gpu01', 'Wrong host/account'


def name(value):
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{5,95}', value), 'Unsafe unique name'
    return value


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base(path):
    account()
    path = own_path(path)
    manifest = read(HERE / 'base-source-sha256.json')
    for file, digest in manifest.items():
        target = own_path(path / file)
        assert sha(target) == digest, 'Frozen base mismatch: ' + file
    sys.path.insert(0, str(path))
    return path, load_module(path / HELPER, 'frozen_rlt_cycle')


class ProcessIdentityChanged(ProcessLookupError):
    """A dynamic process observation cannot be bound to the expected account."""
    def __init__(self, pid, reason, **observed):
        self.identity_context = dict(pid=int(pid), reason=reason, **observed)
        super().__init__(str(self.identity_context))


class ProcessIdentityUncertain(RuntimeError):
    """A registered live generation is not proved exited or safe to control."""
    def __init__(self, pid, reason, **observed):
        self.identity_context = dict(pid=int(pid), reason=reason, **observed)
        super().__init__(str(self.identity_context))


def _proc_contents(directory_fd, filename):
    source = os.open(filename, os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory_fd)
    with os.fdopen(source, 'rb') as stream:
        return stream.read()


def identity(pid):
    # An open proc directory pins this lookup to one PID generation. Directory
    # inode ownership is metadata, not a stable process-UID snapshot: pid_getattr
    # can return default root ownership after its task has disappeared.
    pid = int(pid)
    fd = os.open('/proc/' + str(pid), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    def content(filename):
        return _proc_contents(fd, filename)
    def fields():
        # comm can contain non-UTF8 bytes; only the ASCII fields after ')' matter.
        return content('stat').rsplit(b')', 1)[1].split()
    def uids():
        lines = content('status').splitlines()
        return tuple(map(int, next(line for line in lines if line.startswith(b'Uid:')).split()[1:]))
    try:
        first = fields(); before = uids()
        if before != (UID, UID, UID, UID):
            raise ProcessIdentityChanged(pid, 'foreign_or_changed_process_uid',
                                         observed_uids=list(before), start=int(first[19]))
        digest = hashlib.sha256(content('cmdline')).hexdigest()
        after = uids(); last = fields()
        if before != after or int(first[19]) != int(last[19]):
            raise ProcessIdentityUncertain(pid, 'inconsistent_process_snapshot',
                before_uids=list(before), after_uids=list(after),
                first_start=int(first[19]), last_start=int(last[19]))
        return dict(pid=pid, uid=UID, start=int(last[19]), ppid=int(last[1]),
                    state=last[0].decode('ascii'), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    command_sha256=digest)
    finally:
        os.close(fd)


def alive(row, command=False):
    try:
        current = identity(row['pid'])
    except ProcessIdentityChanged as error:
        if error.identity_context.get('start') == row['start']:
            raise ProcessIdentityUncertain(row['pid'], 'registered_generation_uid_changed',
                                           registered_start=row['start'], observed=error.identity_context) from error
        return False
    except (FileNotFoundError, ProcessLookupError):
        return False
    return (current['state'] not in ('Z', 'X') and
            all(current[k] == row[k] for k in ('uid', 'start', 'boot')) and
            (not command or current['command_sha256'] == row['command_sha256']))


def bind_config(config):
    config = own_path(config)
    cfg = read(config)
    assert cfg['uid'] == UID and cfg['workers_per_gpu'] == 1
    project = own_path(cfg['project'])
    run = own_path(project / 'runs' / cfg['run_id'])
    assert config.is_relative_to(project)
    return config, cfg, project, run
