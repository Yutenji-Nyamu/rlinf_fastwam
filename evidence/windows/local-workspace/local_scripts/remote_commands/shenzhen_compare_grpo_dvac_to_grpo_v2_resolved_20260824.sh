#!/usr/bin/env bash
set -euo pipefail

OLD=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2/resolved.yaml
NEW=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3/resolved.yaml
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

test -f "$OLD"
test -f "$NEW"

"$PY" - "$OLD" "$NEW" <<'PY'
import sys
import yaml

old_path, new_path = sys.argv[1:]
with open(old_path, encoding="utf-8") as f:
    old = yaml.safe_load(f)
with open(new_path, encoding="utf-8") as f:
    new = yaml.safe_load(f)

def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, name))
    elif isinstance(value, list):
        out[prefix] = value
    else:
        out[prefix] = value
    return out

a, b = flatten(old), flatten(new)
keys = sorted(set(a) | set(b))
diffs = [(key, a.get(key, "<MISSING>"), b.get(key, "<MISSING>")) for key in keys if a.get(key, "<MISSING>") != b.get(key, "<MISSING>")]
print(f"old_leaf_count={len(a)} new_leaf_count={len(b)} differing_leaves={len(diffs)}")
for key, left, right in diffs:
    print(f"{key}\tOLD={left!r}\tNEW={right!r}")
PY
