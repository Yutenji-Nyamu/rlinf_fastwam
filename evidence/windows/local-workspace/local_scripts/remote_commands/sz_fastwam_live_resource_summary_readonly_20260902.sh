set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2
PI=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
date '+TIME %Y-%m-%d %H:%M:%S %Z'
find "$RUN" -maxdepth 3 -type f -printf '%p %s\n' | sort | head -200
"$PY" - "$RUN/runtime/resource.csv" "$PI/runtime/resource.csv" <<'PY'
import csv, os, sys
for p in sys.argv[1:]:
    print('RESOURCE', p, 'exists', os.path.exists(p))
    if not os.path.exists(p): continue
    with open(p, newline='') as f: rows=list(csv.DictReader(f))
    print('rows',len(rows),'fields',list(rows[0]) if rows else [])
    for k in (list(rows[0]) if rows else []):
        if k.endswith('_used_mib'):
            vals=[float(r[k]) for r in rows if r.get(k) not in ('',None)]
            print(k,'max',max(vals) if vals else None,'last',vals[-1] if vals else None)
    print('last_ts',rows[-1].get('timestamp') if rows else None)
PY
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
