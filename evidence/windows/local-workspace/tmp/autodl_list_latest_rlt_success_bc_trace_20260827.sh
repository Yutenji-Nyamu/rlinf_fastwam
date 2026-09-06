#!/usr/bin/env bash
set -euo pipefail

run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
echo IDENTITY
hostname
pwd
id -u
date -Is

echo TRACE_FILES
find "$run" -type f -path '*/rlt_dvac/actor_rank*/update_*.npz' -printf '%T@ %s %p\n' | sort -n | tail -n 12

echo CONFIG
find "$run" -maxdepth 4 -type f \( -name 'resolved_config.yaml' -o -name 'config.yaml' \) -printf '%s %p\n' | sort -k2
