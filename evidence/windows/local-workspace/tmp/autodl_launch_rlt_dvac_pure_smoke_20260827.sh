#!/usr/bin/env bash
set -euo pipefail
script=/tmp/autodl_run_rlt_dvac_pure_smoke_20260827.sh
launcher_log=/tmp/rlt_dvac_pure_smoke_launcher_20260827_v2.log
test -f "$script"
test ! -e /root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2
test ! -e /root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2
test ! -e "$launcher_log"
setsid bash "$script" >"$launcher_log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >/tmp/rlt_dvac_pure_smoke_launcher_20260827_v2.pid
echo "SMOKE_LAUNCHER_PID=$pid"
echo "SMOKE_LAUNCHER_LOG=$launcher_log"
