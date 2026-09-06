#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

launch=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime/remote_ogpo_20260807_formal_launch_v1.sh
expected_sha=be32fe047759654056aa80c7bd9e7379cf88f0bbec4ccf9b845a6bb41bb64a52

test -f "$launch"
test "$(sha256sum "$launch" | awk '{print $1}')" = "$expected_sha"
bash -n "$launch"
exec bash "$launch"
