#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime
monitor="$runtime_root/remote_ogpo_20260807_smoke_monitor_v1.sh"
runner="$runtime_root/remote_ogpo_20260807_smoke_run_v1.sh"
monitor_sha=40e65af8e03a3ebaafa1ff322aee99795832bcd34fafc006f80b8accff0c5368
runner_sha=ba3cf334784f4d237febe7edf37bc1104cba997e07351fca1e73e29f4074a9ad

test "$(sha256sum "$monitor" | awk '{print $1}')" = "$monitor_sha"
test "$(sha256sum "$runner" | awk '{print $1}')" = "$runner_sha"
bash -n "$monitor"
bash -n "$runner"
stat --printf='%s\t%y\t%n\n' "$monitor" "$runner"
printf 'MONITOR_SHA256\t%s\n' "$monitor_sha"
printf 'RUN_SHA256\t%s\n' "$runner_sha"
printf '%s\n' OGPO_SMOKE_RUNTIME_SCRIPTS_VERIFIED
