#!/usr/bin/env bash
set -euo pipefail
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
PI05=/data/chenyiteng/results/rlinf-shenzhen/pi05/packets/pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/resolved.yaml
PI0=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/runtime/resolved.yaml
"$VENV/bin/python" - "$PI05" "$PI0" <<'PY'
import json, sys, yaml
a,b=[yaml.safe_load(open(p,encoding="utf-8")) for p in sys.argv[1:]]
def flat(v,p=""):
 out={}
 if isinstance(v,dict):
  for k,x in v.items(): out.update(flat(x,f"{p}.{k}" if p else str(k)))
 else: out[p]=v
 return out
fa,fb=flat(a),flat(b)
rows=[]
for k in sorted(set(fa)|set(fb)):
 if fa.get(k)!=fb.get(k): rows.append({"key":k,"pi05":fa.get(k),"pi0":fb.get(k)})
print(json.dumps(rows,ensure_ascii=False,indent=2))
PY
