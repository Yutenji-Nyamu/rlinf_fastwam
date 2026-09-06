set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin:${PYTHONPATH:-}"
"$venv/bin/python" - <<'PY'
import inspect
import openpi
from openpi.models_pytorch import pi0_pytorch

print("openpi", inspect.getfile(openpi))
print("pi0_pytorch", inspect.getfile(pi0_pytorch))
print("PI0Pytorch", inspect.getfile(pi0_pytorch.PI0Pytorch))
PY
