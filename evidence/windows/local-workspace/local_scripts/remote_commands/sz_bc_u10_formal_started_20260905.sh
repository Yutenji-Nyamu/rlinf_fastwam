set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -u - <<'PY'
import datetime, json, os, re, subprocess, time
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1')
for _ in range(10):
    log_path=run/'driver.log'
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',log_path.read_text(errors='replace')) if log_path.exists() else ''
    errors=[line for line in log.splitlines() if re.search(r'Fatal Python|Traceback \(most recent|OutOfMemoryError|ErrorInitializationFailed|cannot create buffer|pthread_key_create|AssertionError|RuntimeError:',line)]
    wrapper=int((run/'wrapper.pid').read_text().strip())
    proc=Path('/proc')/str(wrapper)
    alive=proc.exists() and proc.stat().st_uid==os.getuid()
    rc=(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None
    csv=run/'resource.csv'
    collecting='Generating Rollout Epochs:' in log
    result={'time':datetime.datetime.now().astimezone().isoformat(),'wrapper_pid':wrapper,'wrapper_alive':alive,
        'exit':rc,'collecting':collecting,'errors':errors[:5],
        'source_head':(run/'runtime/source-head.txt').read_text().strip() if (run/'runtime/source-head.txt').exists() else None,
        'started_at':(run/'started_at.txt').read_text().strip() if (run/'started_at.txt').exists() else None,
        'recent':[line[-500:] for line in log.splitlines() if re.search(r'parameter|namespace|Generating Rollout|ActorGroup|EnvGroup',line)][-5:],
        'resource':csv.read_text().splitlines()[-1] if csv.exists() else None}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if errors or rc is not None or not alive or collecting:
        (run/'runtime/startup-verification.json').write_text(json.dumps(result,indent=2))
        break
    time.sleep(30)
for command in [
    ['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
    ['df','-h','/data'],['free','-h'],
    ['ps','-p','602620,321933,322685','-o','pid,etime,stat,comm'],
]:
    print(subprocess.run(command,capture_output=True,text=True,timeout=20).stdout,flush=True)
PY
