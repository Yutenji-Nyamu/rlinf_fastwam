#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
LOG=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821/curobo_v078_pin_no_deps_retry1.log
PGID=1268695

printf '%s\n' '=== LOG TAIL ==='
tail -n 40 "$LOG"
printf '%s\n' '=== PROCESS GROUP ==='
ps -u "$(id -u)" -o pid=,ppid=,pgid=,stat=,etime=,comm= \
  | awk -v pgid="$PGID" '$3 == pgid'
printf '%s\n' '=== INSTALLED STATE ==='
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import torch

dist = md.distribution("nvidia-curobo")
direct = json.loads((Path(dist._path) / "direct_url.json").read_text())
print({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "curobo_version": dist.version, "curobo_commit": direct["vcs_info"]["commit_id"]})
PY
