set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
RUFF_BIN=/root/autodl-tmp/RLinf/.venv/bin/ruff

cd "$RLT_ROOT"

FILES="
rlinf/algorithms/rlt/__init__.py
rlinf/algorithms/rlt/rollout.py
rlinf/algorithms/rlt/route.py
rlinf/algorithms/rlt/transition.py
rlinf/models/embodiment/openpi/__init__.py
rlinf/models/embodiment/openpi/openpi_action_model.py
rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
tests/unit_tests/test_robotwin_rlt_contract.py
toolkits/rlt/probe_robotwin_rlt_prefix_contract.py
"

# shellcheck disable=SC2086
"$RUFF_BIN" check --fix $FILES
# shellcheck disable=SC2086
"$RUFF_BIN" format $FILES
