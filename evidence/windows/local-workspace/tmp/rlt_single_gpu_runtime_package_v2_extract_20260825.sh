#!/usr/bin/env bash
set -euo pipefail
bundle=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual_runtime_20260825_v2.tar.gz
package_root=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_runtime_package
test -f "$bundle"
tar -xzf "$bundle" -C "$package_root"
chmod 700 "$package_root"/*.sh
for f in "$package_root"/*.sh; do bash -n "$f"; done
sha256sum "$bundle" "$package_root"/*.sh
ps -o pid,ppid,pgid,etimes,cmd -p "$(cat /root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue/queue.pid)"
