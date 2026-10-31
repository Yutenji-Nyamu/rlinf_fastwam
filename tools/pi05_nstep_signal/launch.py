"""Register two prepared per-card lanes before releasing exact RLT drivers."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/rlt_disk_fallback'))
from priority_pair_handoff import handoff, proc, read, save, same, exact_signal


def main():
    D=Path(sys.argv[1]);assert read(D/'cpu-checks.json')['passed']
    owner=proc(os.getpid());save(D/'launch-identity.json',owner)
    reservations={}
    try:
        for g in [6,7]:
            with (D/f'lane-g{g}.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'tools/pi05_nstep_signal/lane.py'),str(D),str(g)],
                      cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL,start_new_session=True)
            deadline=time.time()+30
            while not (D/f'lane-g{g}-identity.json').exists():
                assert child.poll() is None and time.time()<deadline
                time.sleep(.1)
            reservations[str(g)]=dict(identity=read(D/f'lane-g{g}-identity.json'),release=str(D/f'lane-g{g}-released.json'))
        receipt=handoff(D,[6,7],owner,reservations=reservations)
        save(D/'per-card-handoff.json',dict(receipt,reservations=reservations))
    except BaseException:
        for r in reservations.values():
            if same(r['identity']):exact_signal(r['identity'],signal.SIGTERM)
        raise


if __name__=='__main__':
    main()
