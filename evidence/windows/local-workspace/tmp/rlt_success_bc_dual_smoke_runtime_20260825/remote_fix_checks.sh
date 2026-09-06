set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
venv=/root/autodl-tmp/RLinf/.venv/bin
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
echo DIFF_STAT
git diff --stat
echo DIFF_CHECK
git diff --check
echo FORMAT
"$venv/ruff" format rlinf/algorithms/rlt/transition.py rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py tests/unit_tests/test_rlt_dvac_weighting.py
echo LINT
"$venv/ruff" check rlinf/algorithms/rlt/transition.py rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py tests/unit_tests/test_rlt_dvac_weighting.py
echo PYCOMPILE
"$venv/python" -m py_compile rlinf/algorithms/rlt/transition.py rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py tests/unit_tests/test_rlt_dvac_weighting.py
echo PYTEST
"$venv/pytest" -q tests/unit_tests/test_rlt_dvac_weighting.py
echo STATUS
git status --short
echo DIFF
git diff -- rlinf/algorithms/rlt/transition.py rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py tests/unit_tests/test_rlt_dvac_weighting.py
