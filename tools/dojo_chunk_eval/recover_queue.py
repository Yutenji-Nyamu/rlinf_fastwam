# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Bounded recovery of a private queue if its shared policy/device fails."""
import os
import subprocess
import sys
import time
from pathlib import Path
from common import read, save
from run_task import ops


def main(deployment):
    d = Path(deployment).resolve()
    o = ops()
    me = o.proc(os.getpid())
    save(d / 'recovery-identity.json', me)
    p = read(d / 'prepared.json')
    output = Path(p['output'])
    for attempt in range(2):
        identity = read(d / 'identity.json')
        while o.same(identity):
            time.sleep(10)
        if (d / 'complete.json').exists():
            return
        failure = read(d / 'failure.json')
        reason = failure.get('error', '')
        if not any(marker in reason for marker in ('GPU device fault', 'Shared policy exited')):
            return
        if o.gpus()[p['gpu']]['processes']:
            save(d / 'recovery-blocked.json', {'time':time.time(), 'reason':'private GPU did not clear'})
            return
        # Preserve interruption evidence and retry only jobs interrupted by the
        # failed shared service. Normal failures remain recorded/skipped.
        moved = []
        for result in output.glob('tasks/*/result.json'):
            if 'SystemExit: 143' not in (read(result).get('error') or ''):
                continue
            assert not result.parent.is_symlink()
            dest = output / 'interrupted' / f'{result.parent.name}-recovery-{attempt}'
            dest.parent.mkdir(exist_ok=True)
            result.parent.rename(dest)
            driver_log = d / (result.parent.name + '-driver.log')
            if driver_log.exists():
                driver_log.rename(dest / 'driver.log')
            moved.append(str(dest))
        for name in ('policy-ready.json','failure.json','released.json','identity.json','registered.json'):
            (d/name).unlink(missing_ok=True)
        source = Path(__file__).resolve().parent
        with (d/'queue.log').open('a') as log:
            child = subprocess.Popen([sys.executable,'-u','-B',str(source/'parallel_queue.py'),str(d)],
                                     cwd=source,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while not(d/'identity.json').exists():
            assert child.poll() is None
            time.sleep(.2)
        save(d/'registered.json',{'time':time.time(),'identity':read(d/'identity.json'),'gpu':p['gpu'],'no_rlt':True})
        save(d/f'recovery-{attempt}.json',{'time':time.time(),'previous_failure':reason,'retained_interruptions':moved})


if __name__ == '__main__':
    main(sys.argv[1])
