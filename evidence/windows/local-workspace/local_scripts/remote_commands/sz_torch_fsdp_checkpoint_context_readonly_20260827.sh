#!/usr/bin/env bash
set -euo pipefail

PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
$PY - <<'PY'
from pathlib import Path
import inspect
import torch
import torch.distributed.fsdp._optim_utils as optim_utils

path = Path(inspect.getsourcefile(optim_utils))
print(f"torch={torch.__version__}")
print(f"optim_utils={path}")
lines = path.read_text().splitlines()
for line_no in range(1145, 1191):
    print(f"{line_no}: {lines[line_no - 1]}")
PY
printf 'py_spy='; command -v py-spy || true
printf 'gdb='; command -v gdb || true
