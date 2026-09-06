#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
git -C "$repo" push personal codex/rlt-dvac-success-episode-bc
git -C "$repo" status --short
git -C "$repo" rev-parse HEAD
