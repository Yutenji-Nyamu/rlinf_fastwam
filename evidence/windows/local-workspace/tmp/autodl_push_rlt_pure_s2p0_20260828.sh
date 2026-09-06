#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
test "$(git -C "$repo" rev-parse HEAD)" = ff432cbf35e2ea9c3aa805b20fb180a846fc70e5
test -z "$(git -C "$repo" status --short)"
git -C "$repo" push personal HEAD:refs/heads/codex/rlt-dvac-pure-reference-bc
git -C "$repo" ls-remote personal refs/heads/codex/rlt-dvac-pure-reference-bc
