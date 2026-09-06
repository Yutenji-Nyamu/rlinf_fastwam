#!/usr/bin/env bash
set -euo pipefail

RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821

find "$RUN" -maxdepth 4 -type d -name 'global_step_*' -printf '%p\n' | sort -V | tail -n 10
find "$RUN" -maxdepth 3 -type d -printf '%p\n' | sort | head -n 80
