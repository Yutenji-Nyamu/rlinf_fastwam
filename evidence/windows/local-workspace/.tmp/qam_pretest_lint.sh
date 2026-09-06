set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"
files='
examples/embodiment/train_embodied_agent.py
rlinf/config.py
rlinf/models/embodiment/base_policy.py
rlinf/models/embodiment/openpi/__init__.py
rlinf/models/embodiment/openpi/openpi_action_model.py
rlinf/algorithms/qam/__init__.py
rlinf/algorithms/qam/contracts.py
rlinf/algorithms/qam/core.py
rlinf/data/qam_transition_replay.py
rlinf/models/embodiment/modules/qam_critic.py
rlinf/models/embodiment/modules/qam_modules.py
rlinf/workers/actor/fsdp_qam_policy_worker.py
tests/algorithms/qam/oracle/export_official_fixture.py
tests/algorithms/qam/test_core.py
tests/algorithms/qam/test_official_fixture.py
tests/embodiment/test_qam_openpi_adapter.py
tests/embodiment/test_robotwin_qam_contract.py
tests/workers/test_qam_worker_helpers.py
'
new_files='
rlinf/algorithms/qam/__init__.py
rlinf/algorithms/qam/contracts.py
rlinf/algorithms/qam/core.py
rlinf/data/qam_transition_replay.py
rlinf/models/embodiment/modules/qam_critic.py
rlinf/models/embodiment/modules/qam_modules.py
rlinf/workers/actor/fsdp_qam_policy_worker.py
tests/algorithms/qam/oracle/export_official_fixture.py
tests/algorithms/qam/test_core.py
tests/algorithms/qam/test_official_fixture.py
tests/embodiment/test_qam_openpi_adapter.py
tests/embodiment/test_robotwin_qam_contract.py
tests/workers/test_qam_worker_helpers.py
'
if [ -x "$venv/bin/ruff" ]; then
  "$venv/bin/ruff" check $files
  "$venv/bin/ruff" format --check $new_files
else
  echo 'RUFF_NOT_INSTALLED'
fi
