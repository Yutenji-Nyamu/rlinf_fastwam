#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh
fresh=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh
resume=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_resume.sh

bash -n "${monitor}" "${fresh}" "${resume}"
test "$(sha256sum "${monitor}" | cut -d' ' -f1)" = \
  925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b
test "$(sha256sum "${fresh}" | cut -d' ' -f1)" = \
  6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a
test "$(sha256sum "${resume}" | cut -d' ' -f1)" = \
  6494eaeb8cb2e6c1decee07d97798a0aba368ee15b0491faf3ecdfe6a9ff054c
grep -Fq "':(exclude)HANDOFF.md'" "${fresh}"
grep -Fq "':(exclude)HANDOFF.md'" "${resume}"
test ! -e /root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
test ! -e /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
sha256sum "${monitor}" "${fresh}" "${resume}"
printf '%s\n' STAGE2_LAUNCH_SCRIPTS_V3_OK
