#!/usr/bin/env bash
set -euo pipefail

ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
source /etc/profile.d/mihomo-proxy.sh
export PATH=/home/chenyiteng/miniforge3/bin:$PATH
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
unset PYTHONPATH

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
python -m pip show packaging setuptools setuptools-scm vcs-versioning fastwam
python - <<'PY'
from importlib import metadata
from packaging.requirements import Requirement

targets = {"setuptools-scm", "vcs-versioning"}
for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
    name = dist.metadata.get("Name") or "<unknown>"
    for raw in dist.requires or []:
        try:
            dep = Requirement(raw).name.lower().replace("_", "-")
        except Exception:
            continue
        if dep in targets:
            print(f"consumer={name} requirement={raw}")
PY
python -m pip check || true
python - <<'PY'
import fastwam
import torch
print({"fastwam": fastwam.__file__, "torch": torch.__version__, "cuda": torch.version.cuda})
PY
