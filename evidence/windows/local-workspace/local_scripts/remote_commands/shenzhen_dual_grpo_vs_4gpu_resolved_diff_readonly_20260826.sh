#!/usr/bin/env bash
set -euo pipefail
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
OLD=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2/resolved.yaml
NEW=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-control-formal100-2gpu64x4-b1024-eval5-phys45-v1/resolved.yaml
test -s "$OLD"; test -s "$NEW"
"$VENV/bin/python" - "$OLD" "$NEW" <<'PY'
import sys, yaml

def flatten(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(x, list):
        out[prefix] = x
    else:
        out[prefix] = x
    return out

old, new = [flatten(yaml.safe_load(open(p, encoding="utf-8"))) for p in sys.argv[1:]]
missing = object()
for key in sorted(set(old) | set(new)):
    if old.get(key, missing) != new.get(key, missing):
        print(f"{key}\t{old.get(key, '<MISSING>')!r}\t{new.get(key, '<MISSING>')!r}")
PY
