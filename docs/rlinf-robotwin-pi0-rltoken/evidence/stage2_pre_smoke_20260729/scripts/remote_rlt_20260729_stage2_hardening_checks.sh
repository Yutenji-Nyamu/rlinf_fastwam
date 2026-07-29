#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "${repo}"

test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = 6df42bf488ef10d9c7eb2f89584bc5ab7543a08a

"${venv}/bin/python" -m py_compile \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_robotwin_rlt_contract.py \
  toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py \
  toolkits/rlt/audit_robotwin_rlt_stage2_resolved.py
"${venv}/bin/ruff" check \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_robotwin_rlt_contract.py \
  toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py \
  toolkits/rlt/audit_robotwin_rlt_stage2_resolved.py
git diff --check

PYTHONPATH="${repo}:/root/autodl-tmp/RoboTwin_RLinf" \
PYTHONDONTWRITEBYTECODE=1 \
"${venv}/bin/python" -m pytest -q \
  tests/unit_tests/test_robotwin_rlt_contract.py

printf '%s\n' STAGE2_HARDENING_CHECKS_OK
