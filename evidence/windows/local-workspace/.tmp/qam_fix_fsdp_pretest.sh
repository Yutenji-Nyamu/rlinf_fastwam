set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"
"$venv/bin/ruff" format \
  rlinf/models/embodiment/modules/qam_modules.py \
  tests/embodiment/test_qam_openpi_adapter.py
"$venv/bin/ruff" check \
  rlinf/models/embodiment/modules/qam_modules.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  tests/embodiment/test_qam_openpi_adapter.py
git diff --check
"$venv/bin/python" -m pytest -q \
  tests/embodiment/test_qam_openpi_adapter.py
