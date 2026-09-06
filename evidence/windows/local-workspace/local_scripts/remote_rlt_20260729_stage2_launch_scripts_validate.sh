#!/usr/bin/env bash
set -euo pipefail

monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh
fresh=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh
resume=/root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_resume.sh
selftest=/root/autodl-tmp/tmp/rlt_stage2_monitor_selftest_20260729_v2.csv

bash -n "${monitor}" "${fresh}" "${resume}"
test ! -e "${selftest}"
bash "${monitor}" 999999 "${selftest}" 1
awk -F, '
  NR == 1 {expected = NF}
  NF != expected {exit 1}
  END {
    if (NR != 2) {
      exit 2
    }
    print "STAGE2_MONITOR_SELFTEST_OK fields=" expected " rows=" NR
  }
' "${selftest}"

sha256sum "${monitor}" "${fresh}" "${resume}"
test ! -e /root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
test ! -e /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
printf '%s\n' STAGE2_LAUNCH_SCRIPTS_SYNTAX_OK
