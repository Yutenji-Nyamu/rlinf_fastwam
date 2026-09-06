set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
RUFF_BIN=/root/autodl-tmp/RLinf/.venv/bin/ruff

cd "$RLT_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1

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

"$PYTHON_BIN" -B - <<'PY'
from pathlib import Path

files = [
    "rlinf/algorithms/rlt/__init__.py",
    "rlinf/algorithms/rlt/rollout.py",
    "rlinf/algorithms/rlt/route.py",
    "rlinf/algorithms/rlt/transition.py",
    "rlinf/models/embodiment/openpi/__init__.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py",
    "tests/unit_tests/test_robotwin_rlt_contract.py",
    "toolkits/rlt/probe_robotwin_rlt_prefix_contract.py",
]
for file_name in files:
    source = Path(file_name).read_text(encoding="utf-8")
    compile(source, file_name, "exec")
print("AST_COMPILE_OK", len(files))
PY

"$RUFF_BIN" --version
# shellcheck disable=SC2086
"$RUFF_BIN" check $FILES
# shellcheck disable=SC2086
"$RUFF_BIN" format --check $FILES
git diff --check
