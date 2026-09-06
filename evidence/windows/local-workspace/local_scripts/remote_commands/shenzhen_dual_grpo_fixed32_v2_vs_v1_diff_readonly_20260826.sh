#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets
OLD=$ROOT/grpo-control-formal100-2gpu64x4-b1024-eval5-phys45-v1/resolved.yaml
NEW=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/resolved.yaml
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - "$OLD" "$NEW" <<'PY'
import sys, yaml
def flat(v, p=""):
    if isinstance(v, dict):
        out={}
        for k,x in v.items(): out.update(flat(x, f"{p}.{k}" if p else str(k)))
        return out
    return {p:v}
a=flat(yaml.safe_load(open(sys.argv[1], encoding="utf-8")))
b=flat(yaml.safe_load(open(sys.argv[2], encoding="utf-8")))
for k in sorted(set(a)|set(b)):
    if a.get(k) != b.get(k): print(f"{k}: {a.get(k)!r} -> {b.get(k)!r}")
PY
