#!/usr/bin/env bash
set -euo pipefail

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime
bash -n "$runtime/run_foreground.sh"
bash -n "$runtime/launch_background.sh"
test -x /root/autodl-tmp/RLinf/.venv/bin/python
grep -A8 '^timeout --signal=TERM' "$runtime/run_foreground.sh"
printf '%s\n' RLT_RESOURCE_SMOKE_LAUNCHER_SYNTAX_OK
