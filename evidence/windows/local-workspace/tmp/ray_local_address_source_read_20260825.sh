#!/usr/bin/env bash
set -euo pipefail
venv=/root/autodl-tmp/RLinf/.venv
ray_pkg=$($venv/bin/python -c 'import ray,os; print(os.path.dirname(ray.__file__))')
grep -R "address == \"local\"\|address='local'\|address=\"local\"" -n "$ray_pkg/_private" | head -n 80
"$venv/bin/python" - <<'PY'
import inspect
from ray._private import worker
source = inspect.getsource(worker.init)
for i, line in enumerate(source.splitlines(), 1):
    if 'local' in line or 'address' in line and 410 <= i <= 620:
        print(f'{i}: {line}')
PY
