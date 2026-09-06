#!/usr/bin/env bash
set -u

runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
mkdir -p "$runtime_dir"
test -d "$runtime_dir"
printf '%s\n' "$runtime_dir"
