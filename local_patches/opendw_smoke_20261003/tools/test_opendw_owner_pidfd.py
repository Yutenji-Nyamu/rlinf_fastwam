"""CPU-only pidfd regression: signals only freshly created sleep children.

Run with the current RLT Python on SZ3. Imports no Ray, torch, or GPU libraries.
Both the normal dispatch and forced missing-Python-bindings fallback are tested.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


def identity(pid):
    path = Path('/proc')/str(pid)
    fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'uid': path.stat().st_uid, 'start': int(fields[19]), 'state': fields[0]}


def same(expected):
    try:
        actual = identity(expected['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return False
    return actual['state'] not in ('Z', 'X') and all(actual[k] == expected[k] for k in ('pid', 'uid', 'start'))


def exercise(module, label):
    module.pidfd_probe()
    module.H = SimpleNamespace(same=same)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    expected = identity(child.pid)
    assert expected['uid'] == os.getuid()
    catalog = module.Catalog(Path('/unused-cpu-test-catalog'), 'cpu-test')
    try:
        catalog.send(dict(expected, start=expected['start']+1), signal.SIGTERM)
        catalog.send(dict(expected, uid=expected['uid']+1), signal.SIGTERM)
        time.sleep(0.1)
        assert child.poll() is None, 'Mismatched identity was signalled'
        fd = module.pidfd_open(child.pid)
        try:
            assert not os.get_inheritable(fd), 'pidfd must be close-on-exec'
            module.pidfd_send(fd, 0)
        finally:
            os.close(fd)
        catalog.send(expected, signal.SIGTERM)
        assert child.wait(timeout=5) == -signal.SIGTERM
        catalog.send(expected, signal.SIGTERM)  # Dead identities are a safe no-op.
        return {'mode': label, 'wrong_start_untouched': True, 'wrong_uid_untouched': True,
                'exact_child_terminated': True, 'no_gpu_used': True}
    finally:
        if child.poll() is None:
            # Only our exact test child; use the same pidfd path during teardown.
            catalog.send(expected, signal.SIGKILL)
            child.wait(timeout=5)


def main():
    path = Path(__file__).with_name('opendw_smoke_owner.py')
    spec = importlib.util.spec_from_file_location('opendw_owner_pidfd_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = [exercise(module, 'available_python_bindings_or_fallback')]
    old_open = getattr(os, 'pidfd_open', None)
    old_send = getattr(signal, 'pidfd_send_signal', None)
    try:
        os.pidfd_open = None
        signal.pidfd_send_signal = None
        results.append(exercise(module, 'forced_ctypes_syscalls'))
    finally:
        if old_open is None:
            del os.pidfd_open
        else:
            os.pidfd_open = old_open
        if old_send is None:
            del signal.pidfd_send_signal
        else:
            signal.pidfd_send_signal = old_send
    print(json.dumps({'passed': True, 'checks': results}))


if __name__ == '__main__':
    main()
