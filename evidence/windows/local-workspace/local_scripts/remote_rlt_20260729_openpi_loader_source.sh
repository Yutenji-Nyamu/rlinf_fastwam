#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/root/autodl-tmp/RLinf_rlt_pi0_robotwin:/root/autodl-tmp/RoboTwin_RLinf
export PYTHONDONTWRITEBYTECODE=1

/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import inspect
import openpi.training.data_loader as dl

print("MODULE", inspect.getsourcefile(dl))
for name in (
    "create_data_loader",
    "create_torch_data_loader",
    "DataLoaderImpl",
    "TorchDataLoader",
):
    obj = getattr(dl, name, None)
    print(f"=== {name} ===")
    if obj is None:
        print("MISSING")
    else:
        print(inspect.getsource(obj))
PY
