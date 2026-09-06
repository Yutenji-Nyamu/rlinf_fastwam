#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
tar -xf /tmp/rlt_dvac_pure_worker_fix_20260827.tar -C "$repo"
git -C "$repo" diff --check
git -C "$repo" status --short
