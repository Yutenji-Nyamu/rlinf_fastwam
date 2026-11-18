"""Run the requested record first, then distribute the remaining observations over three workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(path):
    job=json.loads(path.read_text())
    root=Path(job['output'])
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    assert gpu in ('4','5')
    workers=[[],[],[]];loads=[0,0,0]
    for record in job['episodes']:
        i=loads.index(min(loads))
        workers[i].append(record);loads[i]+=record['frames']
    procs=[]
    def start(i):
        worker=f'{gpu}-{i}'
        config=path.with_name(f'job-{worker}.json')
        config.write_text(json.dumps(dict(job,worker=worker,episodes=workers[i])))
        with (path.parent/f'worker-{worker}.log').open('a') as log:
            procs.append(subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).with_name('collect.py')),str(config)],stdout=log,stderr=subprocess.STDOUT))
    start(0)
    progress=root/f'progress-g{gpu}-0.json'
    first=root/workers[0][0]['id']/'complete.json'
    while not first.exists() and not progress.exists() and procs[0].poll() is None:
        time.sleep(2)
    if procs[0].poll() not in (None,0):raise RuntimeError('first worker failed to start')
    for i in (1,2):start(i)
    codes=[p.wait() for p in procs]
    incomplete=[r['id'] for r in job['episodes'] if not (root/r['id']/'complete.json').is_file()]
    if incomplete:
        # Retry individual failures once in a fresh, single process, preserving completed frames.
        retry=path.with_name(f'job-retry-{gpu}.json')
        retry.write_text(json.dumps(dict(job,worker=gpu+'-retry',episodes=[r for r in job['episodes'] if r['id'] in incomplete])))
        subprocess.run([sys.executable,'-u','-B',str(Path(__file__).with_name('collect.py')),str(retry)],check=True)
    assert all((root/r['id']/'complete.json').is_file() for r in job['episodes']),codes
    (root/f'complete-slot-{gpu}.json').write_text(json.dumps(dict(episodes=len(job['episodes']),time=time.time())))

if __name__=='__main__':main(Path(sys.argv[1]))
