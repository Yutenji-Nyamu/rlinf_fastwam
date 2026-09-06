set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import datetime,json,re,subprocess,time
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8')
for _ in range(100):
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace') if (run/'driver.log').exists() else '')
    lines=log.splitlines();steps=re.findall(r'Global Step:\s*(\d+)\s*/',log)
    errors=[l for l in lines if any(s in l for s in ('OutOfMemoryError','Fatal Python error','ValueError:','RuntimeError:','Exiting main process'))]
    progress=[l[-350:] for l in lines if any(s in l for s in ('Global Step:','Rollout Epochs:','Saving checkpoint','actor_loss'))]
    code=(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None
    gpu=subprocess.run(['nvidia-smi','-i','6','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
    print(json.dumps({'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'step':int(steps[-1]) if steps else 0,'exit':code,'gpu6':gpu,'progress':progress[-3:],'micro_forward_lines':log.count('Forcing gradient checkpointing to be enabled for Gemma expert model'),'errors':errors[-2:],'log_bytes':len(log)},ensure_ascii=False),flush=True)
    if code is not None:break
    time.sleep(45)
PY
