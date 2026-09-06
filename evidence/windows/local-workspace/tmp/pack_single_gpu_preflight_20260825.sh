#!/usr/bin/env bash
set -euo pipefail
src=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_preflight
out=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_preflight.tar.gz
tar -C "$src" -czf "$out" control.resolved.yaml dvac.resolved.yaml contract_diff.json RESOLVED_SHA256.txt
sha256sum "$out"
du -h "$out"
