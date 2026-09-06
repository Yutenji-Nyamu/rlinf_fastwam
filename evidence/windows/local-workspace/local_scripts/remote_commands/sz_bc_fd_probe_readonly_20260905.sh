#!/usr/bin/env bash
set -eu
id
date -Is
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
/usr/bin/python3 - <<'PY'
import json,subprocess
from pathlib import Path
lines=subprocess.check_output(['ps','-u','chenyiteng','-o','pid=,args='],text=True).splitlines()
for line in lines:
    pid,_,args=line.strip().partition(' ')
    if args.strip()=='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -':
        root=Path('/proc')/pid
        fields=dict(x.split(':',1) for x in (root/'status').read_text().splitlines())
        print(json.dumps({'probe_pid':pid,'open_fds':len(list((root/'fd').iterdir())),
        'rss':fields.get('VmRSS'),'threads':fields.get('Threads'),
        'limits':[x for x in (root/'limits').read_text().splitlines() if 'Max open files' in x]}))
print((Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/env-fd-probe-20260905')/'result.json').exists())
PY
