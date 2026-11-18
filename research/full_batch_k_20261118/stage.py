import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
job=json.loads((root/'job.json').read_text())
out=Path(job['output'])
out.mkdir(parents=True,exist_ok=True)
with (out/'memory.jsonl').open('a',buffering=1) as log:
    p=subprocess.Popen([sys.executable,'-u','-B',str(root/'round_cost.py'),'--job',str(root/'job.json')])
    while p.poll() is None:
        r=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if r.returncode==0:log.write(json.dumps(dict(clock=time.perf_counter(),gib=float(r.stdout.strip())/1024))+'\n')
        time.sleep(1)
    sys.exit(p.returncode)

