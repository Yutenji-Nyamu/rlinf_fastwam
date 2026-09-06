#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
cd "${repo}"

test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = 6df42bf488ef10d9c7eb2f89584bc5ab7543a08a
test "$(sha256sum rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py | cut -d' ' -f1)" = \
  4832f0293bacd29c8dab29d9b828aa83b4a346c3247a8d24026a441171d4129c
test "$(sha256sum tests/unit_tests/test_robotwin_rlt_contract.py | cut -d' ' -f1)" = \
  9e8cd2b0515dd0f7b722fedd84b4c3770ad8fb3132b936f88da9fd6bf5549091
test "$(sha256sum examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml | cut -d' ' -f1)" = \
  426c09e2d9b036c566560124059917e9c22059457e75035e924fb678f6018637
test ! -e toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py

tracked_changes="$(
  git status --short \
    rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
    tests/unit_tests/test_robotwin_rlt_contract.py \
    examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml
)"
test -z "${tracked_changes}"

printf '%s\n' STAGE2_HARDENING_UPLOAD_GUARD_OK
