#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RUN=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
CUROBO_REV=d64c4b005459db10c5dd867d8b30a87d5bda9bdb
source "$VENV/bin/activate"

exec > >(tee "$RUN/scikit_image_install.log") 2>&1

"$VENV/bin/python" - "$CUROBO_REV" <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import sys
import torch
assert torch.__version__ == "2.11.0+cu129"
dist = md.distribution("nvidia-curobo")
direct = json.loads((Path(dist._path) / "direct_url.json").read_text())
assert direct["vcs_info"]["commit_id"] == sys.argv[1]
PY

uv pip install --python "$VENV/bin/python" 'scikit-image==0.26.0'

"$VENV/bin/python" - "$CUROBO_REV" <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import sys
import torch
import skimage
assert torch.__version__ == "2.11.0+cu129"
dist = md.distribution("nvidia-curobo")
direct = json.loads((Path(dist._path) / "direct_url.json").read_text())
assert direct["vcs_info"]["commit_id"] == sys.argv[1]
print({"torch": torch.__version__, "curobo": dist.version, "scikit_image": skimage.__version__, "tifffile": md.version("tifffile"), "lazy_loader": md.version("lazy-loader")})
PY

set +e
uv pip check --python "$VENV/bin/python"
pip_check_rc=$?
set -e
printf 'uv_pip_check_exit=%s\n' "$pip_check_rc"
printf '%s\n' 'R1_CUROBO_DECLARED_DEPENDENCY_OK'
