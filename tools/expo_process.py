"""Minimal pidfd compatibility, copied from the validated WM common helper.

Every caller still verifies boot/UID/PID/start before and after fd creation.
Missing Python bindings use the Linux x86_64 LP64 ABI, never a bare-PID signal.
"""
import os
from pathlib import Path
import platform
import signal
import sys


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


def pidfd_probe():
    """CPU-only capability check: fdinfo identity and signal 0 to self.

    This checks both fallback operations before any resource mutation. Signal
    zero performs the kernel permission/liveness check without delivering one.
    """
    import ctypes
    if (sys.platform != 'linux' or platform.machine() != 'x86_64' or
            ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8):
        raise RuntimeError('EXPO process owners require Linux x86_64 LP64')
    pid = os.getpid()
    fd = pidfd_open(pid)
    try:
        info = (Path('/proc/self/fdinfo') / str(fd)).read_text()
        fields = dict(line.split(':', 1) for line in info.splitlines() if ':' in line)
        if int(fields.get('Pid', '-1').strip()) != pid:
            raise RuntimeError('pidfd fdinfo PID differs from the probe process')
        if os.get_inheritable(fd):
            raise RuntimeError('pidfd must not be inherited by child processes')
        pidfd_send(fd, 0)
        return dict(ok=True, platform='Linux x86_64 LP64', fdinfo_pid_verified=True,
                    signal_zero_verified=True,
                    open_backend='python' if callable(getattr(os, 'pidfd_open', None)) else 'syscall434',
                    send_backend='python' if callable(getattr(signal, 'pidfd_send_signal', None)) else 'syscall424')
    finally:
        os.close(fd)
