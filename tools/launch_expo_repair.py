"""One-shot launch receipt; all resource mutations belong to the detached owner."""
import argparse
import fcntl
import os
import socket
import subprocess
import time

from expo_formal_owner import ROOT, SOURCE, RLT_PY, diagnostic_command, verify_inputs
from expo_formal_resources import previous_owner_released
from expo_smoke_owner import atomic, identity, owned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('diagnostic', 'formal'), default='diagnostic')
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    with (ROOT / 'launch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not any((ROOT / name).exists() for name in ('launch-intent.json', 'owner.json', 'final.json'))
        verify_inputs()
        previous = previous_owner_released()
        if args.mode == 'diagnostic':
            diagnostic_command(ROOT / 'diagnostic.json')
            assert not (ROOT / 'ready.json').exists(), 'Diagnostic gate needs a fresh ready receipt'
            (ROOT / 'controller-heartbeat').touch()
        else:
            assert (ROOT / 'run/checkpoint-latest.pt').is_file()
        command = [RLT_PY, '-X', 'faulthandler', '-u', '-B',
            str(SOURCE / 'tools/expo_formal_owner.py'), '--mode', args.mode]
        intent = dict(time=time.time(), command=command, previous_owner_receipts=previous,
            root=str(ROOT), mode=args.mode)
        atomic(ROOT / 'launch-intent.json', intent)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONFAULTHANDLER='1',
            CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (ROOT / 'owner.log').open('xb') as log:
            process = subprocess.Popen(command, cwd=SOURCE, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        anchor = identity(process.pid)
        assert owned(anchor), 'Owner exited before launch identity registration'
        receipt = dict(**intent, owner=anchor)
        atomic(ROOT / 'launched.json', receipt)
        print(__import__('json').dumps(receipt))


if __name__ == '__main__':
    main()
