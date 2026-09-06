#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
staging=/root/autodl-tmp/qam_formal_patch_20260731

test "$(git -C "$repo" rev-parse HEAD)" = \
  ced8672f322187b71939bd2859842619c6284d05
expected_dirty=$(cat <<'EOF'
 M examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml
 M rlinf/config.py
 M rlinf/workers/actor/fsdp_qam_policy_worker.py
 M tests/workers/test_qam_worker_helpers.py
EOF
)
actual_dirty=$(git -C "$repo" status --porcelain=v1)
test -z "$actual_dirty" || test "$actual_dirty" = "$expected_dirty"

printf '%s  %s\n' \
  c3373753c1d2d3867cd155e7dec8e8f1802186ce21e097e77781614b47bd6670 \
  "$staging/config.py" \
  2e1572031d15befc07e73f9d28462ee1c7cdde550ab4fddad121b411785d4096 \
  "$staging/fsdp_qam_policy_worker.py" \
  0aca13bfd8b24c4f08dc867599c9cedc55f0be7c379f822a661ab71a626b112d \
  "$staging/robotwin_adjust_bottle_qam_openpi.yaml" \
  b5eebf872b6621e038f10f466a7bfd2fd625dc65fd331669269a0a4a917134a7 \
  "$staging/test_qam_worker_helpers.py" \
  | sha256sum -c -

install -m 0644 \
  "$staging/config.py" \
  "$repo/rlinf/config.py"
install -m 0644 \
  "$staging/fsdp_qam_policy_worker.py" \
  "$repo/rlinf/workers/actor/fsdp_qam_policy_worker.py"
install -m 0644 \
  "$staging/robotwin_adjust_bottle_qam_openpi.yaml" \
  "$repo/examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml"
install -m 0644 \
  "$staging/test_qam_worker_helpers.py" \
  "$repo/tests/workers/test_qam_worker_helpers.py"

git -C "$repo" status --short
git -C "$repo" diff --check
git -C "$repo" diff --stat
