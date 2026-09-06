#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1/analyze_shenzhen_dvac_observation.py
PYTHON=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1
EXPECTED_SHA=009b4ea41ab9ebc974b778e46156ee5fc4f0bf843e0fd703d3051715675ff152

test -f "$SCRIPT"
test ! -e "$OUTPUT"
ACTUAL_SHA=$(sha256sum "$SCRIPT" | awk '{print $1}')
test "$ACTUAL_SHA" = "$EXPECTED_SHA"
printf 'SCRIPT_BYTES=%s\nSCRIPT_SHA256=%s\n' "$(stat -c %s "$SCRIPT")" "$ACTUAL_SHA"
"$PYTHON" -m py_compile "$SCRIPT"
"$PYTHON" "$SCRIPT" --help >/dev/null
printf 'REMOTE_PYCOMPILE_AND_HELP_OK\n'

"$PYTHON" - <<'PY'
import json
from pathlib import Path

manifest = Path('/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2/run_manifest.json')
data = json.loads(manifest.read_text(encoding='utf-8'))
print(f"FASTWAM_ACTION_NUM_TRAIN_TIMESTEPS={data.get('action_num_train_timesteps')}")
print(f"FASTWAM_LEGACY_NUM_TRAIN_TIMESTEPS={data.get('num_train_timesteps')}")
PY

printf 'UPLOAD_VERIFY_OK\n'
