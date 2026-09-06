#!/usr/bin/env bash
set -euo pipefail

bundle=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual_runtime_20260825.tar.gz
package_root=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_runtime_package

test -f "$bundle"
test ! -e "$package_root"
mkdir -p "$package_root"
tar -xzf "$bundle" -C "$package_root"
chmod 700 "$package_root"/*.sh

for f in "$package_root"/*.sh; do
  bash -n "$f"
done

sha256sum "$bundle" "$package_root"/*.sh
ls -l "$package_root"

# Confirm that each process-specific Ray temporary directory controls both the
# session root and the current-cluster address file used by address='auto'.
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
venv=/root/autodl-tmp/RLinf/.venv
PYTHONPATH="$repo" RAY_TMPDIR=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_fresh480_20260825_v1/runtime/ray_tmp \
  "$venv/bin/python" - <<'PY'
import inspect
from ray._private import services, utils
print('ray_temp_dir=', utils.get_ray_temp_dir())
print('address_file=', services.get_ray_address_file())
print('address_lookup_source=', inspect.getsource(services.get_ray_address_from_environment))
PY
