#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - <<'PY'
import datetime,json,os,subprocess
from pathlib import Path
d={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'env':{}}
for name,pid in [('bc',2143761),('dvac',2224188),('grpo_rank0',603555),('grpo_rank1',603565)]:
    p=Path('/proc')/str(pid);r={'pid':pid,'exists':p.exists()}
    if p.exists():
        assert p.stat().st_uid==os.getuid()
        try:
            r['smaps_rollup']={k:v.strip() for k,v in (l.split(':',1) for l in (p/'smaps_rollup').read_text().splitlines() if ':' in l)}
            r['status']={k:v.strip() for k,v in (l.split(':',1) for l in (p/'status').read_text().splitlines()) if k in ('Name','VmRSS','RssAnon','RssFile','RssShmem','VmSwap','Threads')}
            r['fd_count']=len(list((p/'fd').iterdir()))
        except OSError as e:r['error']=str(e)
    d['env'][name]=r
d['meminfo']={k:v.strip() for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemAvailable','SwapTotal','SwapFree']}
d['vmstat']=subprocess.run(['vmstat','1','2'],capture_output=True,text=True).stdout
d['memory_pressure']=Path('/proc/pressure/memory').read_text()
print(json.dumps(d))
PY
