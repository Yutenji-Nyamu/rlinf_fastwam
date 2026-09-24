"""Two independent GPU queues. No Ray operations; never signals unrelated PIDs."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from run_inference import save


def validate_batch(path):
    import cv2
    import numpy as np
    cfg = json.loads(Path(path).read_text()); out = Path(cfg['output'])
    done = json.loads((out/'done.json').read_text())
    eps = json.loads((out/'episodes.json').read_text())
    assert done['episodes'] == len(eps) == 16
    assert len({e['actual'] for e in eps}) == 16, 'Native seed fallback duplicated episodes'
    traces = sorted(out.glob('query_*.npz'))
    assert traces
    for trace in traces:
        with np.load(trace) as data:
            assert data['dv'].shape == (16,50) and np.isfinite(data['dv']).all()
            assert data['z_endpoint'].shape == (16,10,50,14)
            assert np.all(data['executed_mask'] <= data['submitted_mask'])
    assert any(float(np.load(p)['dv'].max()) > 0 for p in traces), 'All-zero DV'
    for e in eps:
        cap = cv2.VideoCapture(str(out/e['video']))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok, _ = cap.read(); cap.release()
        assert ok and count == e['frames'] and count >= 2
    return done


def batch(path, gpu, environment, script):
    cfg = json.loads(Path(path).read_text()); out = Path(cfg['output'])
    if (out/'done.json').exists():
        return validate_batch(path)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'started.json').exists():
        raise RuntimeError(f'Partial batch needs explicit retry in a fresh output: {out}')
    cfg['gpu'] = gpu
    assigned = out/'assigned.json'; save(assigned,cfg)
    env = os.environ.copy();env.update(environment);env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env.pop('RAY_ADDRESS',None)
    with (out/'run.log').open('x') as log:
        child = subprocess.Popen([sys.executable,'-u','-B',str(script),str(assigned)],
             cwd=environment['REPO_PATH'],env=env,stdin=subprocess.DEVNULL,
             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'process.json',{'pid':child.pid,'gpu':gpu,'time':time.time()})
        try:
            rc = child.wait(timeout=max(1200,cfg['step_limit']*3))
        except subprocess.TimeoutExpired:
            # The process is our unreaped child in its own session; this group is owned.
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            save(out/'timeout.json',{'time':time.time(),'pid':child.pid})
            raise RuntimeError('Batch timeout: '+str(out))
    if rc:
        raise RuntimeError(f'Batch failed exit={rc}: {out}')
    return validate_batch(path)


def main(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    env=json.loads((root/'environment.json').read_text());script=Path(__file__).with_name('run_inference.py')
    assert m['gpus']==[6,7]
    # flock guards against duplicate queue controllers on the same output tree.
    import fcntl
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(root/'controller.json',{'pid':os.getpid(),'time':time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(batch,p,g,env,script) for p,g in zip(m['smoke'],m['gpus'])]
        results=[f.result() for f in futures]
    save(root/'smoke-passed.json',{'time':time.time(),'batches':results})
    remaining=[p for p in m['configs'] if not (Path(json.loads(Path(p).read_text())['output'])/'done.json').exists()]
    mutex=threading.Lock()
    def worker(gpu):
        while True:
            with mutex:
                if not remaining:return
                path=remaining.pop(0)
            save(root/f'gpu{gpu}-current.json',{'time':time.time(),'config':path})
            try:
                batch(path,gpu,env,script)
            except Exception as exc:
                cfg=json.loads(Path(path).read_text()); dest=Path(cfg['output'])
                save(dest/'batch-failed.json',{'time':time.time(),'error':str(exc),'attempted_episodes':16})
                log=(dest/'run.log').read_text(errors='replace') if (dest/'run.log').exists() else ''
                if any(s in log for s in ('CUDA out of memory','device-side assert','illegal memory access')):
                    raise
    save(root/'formal-started.json',{'time':time.time(),'remaining_batches':len(remaining)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,g) for g in m['gpus']]
        for f in futures:f.result()
    subprocess.run([sys.executable,str(Path(__file__).with_name('rank_curves.py')),str(root)],check=True)
    done=list((root/'batches').glob('*/done.json'));failed=list((root/'batches').glob('*/batch-failed.json'))
    save(root/'complete.json',{'time':time.time(),'completed_batches':len(done),'failed_batches':len(failed)})


if __name__=='__main__':
    main(sys.argv[1])
