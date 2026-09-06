PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -u - <<'PY'
import datetime,json,re,time
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7')
for _ in range(120):
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace'))
    lines=log.splitlines()
    interesting=[line for line in lines if re.search(r'Step|Epoch|Update|Eval|success|checkpoint|Saved|fatal|Error|Traceback|parameters|rollout',line,re.I)]
    errors=[line[-1000:] for line in lines if re.search(r'Fatal Python|Traceback \(most recent|OutOfMemoryError|ErrorInitializationFailed|cannot create buffer|pthread_key_create|AssertionError|RuntimeError:',line)]
    csv=run/'resource.csv'
    ckpts=[str(p.relative_to(run)) for p in run.glob('**/learner.pt')]
    rc=(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None
    print(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'exit':rc,
        'recent':[line[-550:] for line in interesting[-6:]],'errors':errors[:3],
        'resource':csv.read_text().splitlines()[-1] if csv.exists() else None,
        'learner_checkpoints':ckpts}),flush=True)
    if rc is not None:
        break
    time.sleep(45)
PY
