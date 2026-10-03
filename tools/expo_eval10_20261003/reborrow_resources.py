"""New EXPO eval10 borrowing cycle, using the byte-identical validated bridge."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

TRAIN_ROOT = Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002')
ROOT = TRAIN_ROOT.parent / 'eval10-continuation-20261003'
SOURCE = TRAIN_ROOT / 'source'
CYCLE = ROOT / 'rlt-cycle-expo-eval10-20261003-v1'
PREVIOUS = TRAIN_ROOT / 'rlt-cycle-expo-turn-switch-repair-20261002-v1'


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_previous(current, final, release, status):
    require(current.get('status') == 'RLT_RESTORED'
            and final.get('gpu_released') is True and final.get('rlt_first_rounds_verified') is True,
            'Previous EXPO owner has not completed RLT return')
    require(release.get('cycle_id') == PREVIOUS.name and release.get('all_workers_stopped') is True,
            'Previous release receipt differs')
    require(status.get('cycle_id') == PREVIOUS.name and status.get('all_first_rounds_verified') is True,
            'Previous RLT first-round receipt differs')


def previous_owner_released():
    from expo_smoke_owner import owned
    current = read(TRAIN_ROOT / 'current.json')
    final = read(TRAIN_ROOT / 'final.json')
    release = read(TRAIN_ROOT / 'release.json')
    status = read(TRAIN_ROOT / 'rlt-status.json')
    validate_previous(current, final, release, status)
    require(not owned(current['owner']), 'Previous resource owner is still alive')
    require(not owned(read(TRAIN_ROOT / 'owner.json')['owner']), 'Original owner identity remains alive')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'stop', 'resume', 'status'))
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    require(os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02', 'Wrong resource host/UID')
    sys.path.insert(0, str(SOURCE / 'tools'))
    import expo_rlt_switch as bridge
    from expo_formal_resources import pin_signals
    from expo_process import pidfd_probe
    pidfd_probe()
    load = bridge.load_helper
    bridge.load_helper = lambda path: pin_signals(load(path))
    if args.action == 'status':
        result = bridge.frozen(CYCLE).status(CYCLE)
    else:
        CYCLE.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(not CYCLE.is_symlink() and CYCLE.resolve().is_relative_to(ROOT.resolve()), 'New cycle path escaped')
        with (CYCLE / 'operation.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'prepare':
                previous_owner_released()
                result = bridge.prepare(CYCLE, PREVIOUS)
            elif args.action == 'stop':
                result = bridge.stop_with_latest_valid(CYCLE)
            else:
                require(args.release is not None and args.release.resolve(strict=True)
                        == (ROOT / 'release.json').resolve(strict=True), 'Wrong new-cycle release path')
                result = bridge.frozen(CYCLE).resume(CYCLE, args.release)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
