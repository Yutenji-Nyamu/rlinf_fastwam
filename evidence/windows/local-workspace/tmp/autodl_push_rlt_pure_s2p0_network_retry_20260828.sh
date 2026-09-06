#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
(
  source /etc/network_turbo
  git -C "$repo" push personal HEAD:refs/heads/codex/rlt-dvac-pure-reference-bc
  git -C "$repo" ls-remote personal refs/heads/codex/rlt-dvac-pure-reference-bc
)
