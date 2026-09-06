#!/usr/bin/env bash
set -euo pipefail
audit=/root/autodl-tmp/experiment_exports/rlt_matched_width_formal480_v3_preflight_20260826
python=/root/autodl-tmp/RLinf/.venv/bin/python

"$python" - "$audit/historical_rlt_resume250_to480_resolved.yaml" "$audit/control_resolved.yaml" <<'PY'
import sys, yaml

def flatten(x, p=''):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items(): out.update(flatten(v, f'{p}.{k}' if p else str(k)))
    elif isinstance(x, list):
        for i, v in enumerate(x): out.update(flatten(v, f'{p}[{i}]'))
    else: out[p] = x
    return out

old = flatten(yaml.safe_load(open(sys.argv[1], encoding='utf-8')))
new = flatten(yaml.safe_load(open(sys.argv[2], encoding='utf-8')))
diff = [(k, old.get(k, '<MISSING>'), new.get(k, '<MISSING>')) for k in sorted(old.keys() | new.keys()) if old.get(k, '<MISSING>') != new.get(k, '<MISSING>')]
print('HISTORICAL_CONTROL_DIFF_COUNT', len(diff))
for row in diff: print(repr(row))
PY

echo CONFIG_SHA256
sha256sum "$audit"/*resolved.yaml
