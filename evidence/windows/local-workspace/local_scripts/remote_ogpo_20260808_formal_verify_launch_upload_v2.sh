#!/usr/bin/env bash
set -euo pipefail

launch=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime/remote_ogpo_20260808_formal_launch_v2.sh
expected_sha=d94dbaeb955460964d325b7be6a6f3ade5af0b3ea32e1baa2603695ceee20158

test -f "$launch"
test "$(sha256sum "$launch" | awk '{print $1}')" = "$expected_sha"
bash -n "$launch"
printf 'LAUNCH_SHA256\t%s\n' "$expected_sha"
printf '%s\n' OGPO_FORMAL_V2_LAUNCH_UPLOAD_VERIFIED
