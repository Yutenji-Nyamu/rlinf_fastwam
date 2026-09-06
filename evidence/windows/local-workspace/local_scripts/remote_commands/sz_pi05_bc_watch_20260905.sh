set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -u - <<'PY'
import csv,datetime,json,re,time
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1')
started=time.monotonic()
while time.monotonic()-started<5400:
 log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace'))
 errors=[s for s in log.splitlines() if re.search(r'Fatal Python|Traceback \(most recent|OutOfMemoryError|CUDA out of memory|ErrorInitializationFailed|cannot create buffer|pthread_key_create|AssertionError|RuntimeError:',s)]
 marks=[s.strip() for s in log.splitlines() if re.search(r'Global Step|Generating Rollout Epochs|Training Epoch|Evaluating|Saving checkpoint|Saved checkpoint',s)]
 rows=list(csv.DictReader((run/'resource.csv').open()))
 valid=[r for r in rows if r['gpu6_used_mib'].isdigit()]
 last=rows[-1]
 result={'time':datetime.datetime.now().astimezone().isoformat(),'stage':marks[-2:],
   'gpu6_gib':round(int(last['gpu6_used_mib'])/1024,2) if last['gpu6_used_mib'].isdigit() else None,
   'gpu6_peak_gib':round(max(int(r['gpu6_used_mib']) for r in valid)/1024,2),
   'mem_available_gib':round(int(last['host_mem_available_kib'])/1024**2,1),
   'env_fd':last['env_open_fds'],'errors':errors[:3],
   'exit_code':(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None}
 print(json.dumps(result),flush=True)
 if result['exit_code'] is not None or errors:break
 time.sleep(30)
PY
