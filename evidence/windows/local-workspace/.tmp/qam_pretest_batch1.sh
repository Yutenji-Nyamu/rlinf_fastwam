set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"
echo '=== git diff check ==='
git diff --check
echo '=== compile ==='
"$venv/bin/python" -m compileall -q \
  rlinf/algorithms/qam \
  rlinf/data/qam_transition_replay.py \
  rlinf/models/embodiment/modules/qam_critic.py \
  rlinf/models/embodiment/modules/qam_modules.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/algorithms/qam \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
echo '=== qam pytest ==='
"$venv/bin/python" -m pytest -q \
  tests/algorithms/qam/test_core.py \
  tests/algorithms/qam/test_official_fixture.py \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
