set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import datetime, json, re, subprocess, time
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1')
for _ in range(62):
    raw=(run/'driver.log').read_text(errors='replace') if (run/'driver.log').exists() else ''
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',raw)
    lines=log.splitlines()
    steps=re.findall(r'Global Step:\s*(\d+)\s*/',log)
    errors=[l for l in lines if any(s in l for s in ['OutOfMemoryError','Fatal Python error','ValueError:','RuntimeError:','Exiting main process'])]
    progress=[l[-600:] for l in lines if any(s in l for s in ['Online BC DVAC:','Global Step:','Updating policy','Training Epochs','Training epochs','Rollout Epochs:','Eval Env Step:','Rollout Env Step:','Epoch:','Saving checkpoint','actor_loss'])]
    p=subprocess.run(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    exit_code=(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None
    print(json.dumps({'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'step':int(steps[-1]) if steps else 0,'exit':exit_code,'gpu':p.stdout.strip(),'progress':progress[-4:],'errors':errors[-2:],'log_bytes':len(raw)},ensure_ascii=False),flush=True)
    if exit_code is not None:break
    time.sleep(45)
PY
