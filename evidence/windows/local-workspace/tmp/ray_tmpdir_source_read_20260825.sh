#!/usr/bin/env bash
set -euo pipefail
venv=/root/autodl-tmp/RLinf/.venv
RAY_TMPDIR=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_fresh480_20260825_v1/runtime/ray_tmp \
"$venv/bin/python" - <<'PY'
import inspect
from ray._private import services, utils
for module in (services, utils):
    print('MODULE', module.__file__)
    for name in dir(module):
        if 'temp' in name.lower() or 'address' in name.lower():
            print(name)
for name in ('get_ray_address_file', 'get_ray_address_from_environment', 'find_gcs_addresses', 'find_bootstrap_address', 'get_default_ray_temp_dir', 'get_user_temp_dir'):
    obj = getattr(services, name, None) or getattr(utils, name, None)
    if obj is not None:
        print('SOURCE', name)
        print(inspect.getsource(obj))
PY

ray_pkg=$($venv/bin/python -c 'import ray,os; print(os.path.dirname(ray.__file__))')
grep -R "RAY_TMPDIR\|ray_current_cluster" -n "$ray_pkg/_private" | head -n 80
