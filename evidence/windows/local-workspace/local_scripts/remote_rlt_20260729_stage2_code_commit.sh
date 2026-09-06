#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "${repo}"

test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = \
  6df42bf488ef10d9c7eb2f89584bc5ab7543a08a

expected_status=' M examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml
 M rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
 M tests/unit_tests/test_robotwin_rlt_contract.py
?? toolkits/rlt/audit_robotwin_rlt_stage2_resolved.py
?? toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py
?? toolkits/rlt/validate_robotwin_rlt_stage1_artifact.py'
actual_status="$(git status --short)"
test "${actual_status}" = "${expected_status}"

"${venv}/bin/python" -m py_compile \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_robotwin_rlt_contract.py \
  toolkits/rlt/audit_robotwin_rlt_stage2_resolved.py \
  toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py \
  toolkits/rlt/validate_robotwin_rlt_stage1_artifact.py
"${venv}/bin/ruff" check \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_robotwin_rlt_contract.py \
  toolkits/rlt/audit_robotwin_rlt_stage2_resolved.py \
  toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py \
  toolkits/rlt/validate_robotwin_rlt_stage1_artifact.py
PYTHONPATH="${repo}:/root/autodl-tmp/RoboTwin_RLinf" \
PYTHONDONTWRITEBYTECODE=1 \
"${venv}/bin/python" -m pytest -q \
  tests/unit_tests/test_robotwin_rlt_contract.py
git diff --check

git add -- \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_robotwin_rlt_contract.py \
  toolkits/rlt/audit_robotwin_rlt_stage2_resolved.py \
  toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py \
  toolkits/rlt/validate_robotwin_rlt_stage1_artifact.py
git diff --cached --check
git commit -m "fix(rlt): bind Stage 1 artifact before Stage 2"

test -z "$(git status --short)"
git show --stat --oneline --decorate --no-renames HEAD
