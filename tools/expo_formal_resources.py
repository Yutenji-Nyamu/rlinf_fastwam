"""Exact SZ2 RLT handoff; strictly validate the latest complete checkpoint.

The existing generic bridge and frozen helper remain byte-identical. Only the
loaded helper's process-tree signals are wrapped with identity-pinned pidfds.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import types

import expo_rlt_switch as bridge
from expo_process import pidfd_open, pidfd_probe, pidfd_send
from expo_smoke_owner import owned

ROOT = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002')
PREVIOUS = ROOT.parent / 'formal-turn-switch-20261001/rlt-cycle-expo-turn-switch-20261001-v1'
CYCLE = ROOT / 'rlt-cycle-expo-turn-switch-repair-20261002-v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin_signals(module):
    """Only a freshly verified original helper process-tree row can be killed."""
    original_tree = module.process_tree
    anchors = {}

    def process_tree(roots):
        result = original_tree(roots)
        anchors.update(result)
        return result

    def safe_signal(pid, sig):
        assert sig in (signal.SIGTERM, signal.SIGKILL) and pid in anchors
        expected = anchors[pid]
        assert expected['uid'] == 20001
        if not module.same(expected):
            return
        try:
            fd = pidfd_open(pid)
        except ProcessLookupError:
            return
        try:
            if module.same(expected):
                try:
                    pidfd_send(fd, sig)
                except ProcessLookupError:
                    pass
        finally:
            os.close(fd)

    module.process_tree = process_tree
    module.os = types.SimpleNamespace(**{**os.__dict__, 'kill': safe_signal})
    return module


def previous_owner_released():
    previous = PREVIOUS.parent
    current = read(previous / 'current.json')
    final = read(previous / 'final.json')
    assert current['status'] == 'RLT_RESTORED'
    assert final['gpu_released'] and final['rlt_first_rounds_verified']
    assert not owned(read(previous / 'owner.json')['owner'])
    release = read(previous / 'release.json')
    assert release['cycle_id'] == PREVIOUS.name and release['all_workers_stopped']
    return {name: sha(previous / name) for name in ('current.json', 'final.json', 'release.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'stop', 'resume', 'status'))
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    pidfd_probe()
    load = bridge.load_helper
    bridge.load_helper = lambda path: pin_signals(load(path))
    if args.action == 'status':
        result = bridge.frozen(CYCLE).status(CYCLE)
    else:
        CYCLE.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (CYCLE / 'operation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'prepare':
                previous_owner_released()
                result = bridge.prepare(CYCLE, PREVIOUS)
            elif args.action == 'stop':
                result = bridge.stop_with_latest_valid(CYCLE)
            else:
                assert args.release and args.release.resolve(strict=True) == (ROOT / 'release.json').resolve(strict=True)
                result = bridge.frozen(CYCLE).resume(CYCLE, args.release)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
