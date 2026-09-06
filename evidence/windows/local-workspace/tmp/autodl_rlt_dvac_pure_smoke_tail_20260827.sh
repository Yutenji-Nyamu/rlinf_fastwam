#!/usr/bin/env bash
set -euo pipefail
runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2/runtime
tail -n 220 "$runtime/foreground.log"
