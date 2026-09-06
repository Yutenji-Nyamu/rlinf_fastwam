set -euo pipefail
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'DIFF_STAT\n'
git diff --stat
printf 'DIFF_CHECK\n'
git diff --check
printf 'RUFF\n'
/root/autodl-tmp/RLinf/.venv/bin/ruff check \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py
printf 'PYTEST\n'
/root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlt_dvac_weighting.py
printf 'STATUS\n'
git status --short
