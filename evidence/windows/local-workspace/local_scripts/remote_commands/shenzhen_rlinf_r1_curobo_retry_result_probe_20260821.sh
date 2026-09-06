#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
LOG=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821/curobo_v078_pin_no_deps_retry1.log

printf '%s\n' '=== LOG TAIL ==='
test -s "$LOG"
tail -n 160 "$LOG"

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

printf '%s\n' '=== OWNED BUILD PROCESSES ==='
ps -u "$(id -u)" -o pid=,ppid=,pgid=,stat=,etime=,args= \
  | grep -E 'd64c4b|uv pip install|ninja|nvcc' || true
