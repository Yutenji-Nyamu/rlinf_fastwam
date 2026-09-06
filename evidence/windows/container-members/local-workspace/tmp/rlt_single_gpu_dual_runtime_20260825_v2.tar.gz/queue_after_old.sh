#!/usr/bin/env bash
set -euo pipefail

package_root=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_runtime_package
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
old_pid=106844
old_runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime

control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_fresh480_20260825_v1
control_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_fresh480_20260825_v1/runtime
control_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control
control_name=robotwin_adjust_bottle_rlt_single_gpu_control_fresh480_v1

dvac_run=/root/autodl-tmp/experiments/rlt_single_gpu_teacher_dvac_w05to15_fresh480_20260825_v1
dvac_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_teacher_dvac_w05to15_fresh480_20260825_v1/runtime
dvac_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_teacher_dvac_w05to15_fresh480
dvac_name=robotwin_adjust_bottle_rlt_single_gpu_teacher_dvac_w05to15_fresh480_v1

queue_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue
mkdir -p "$queue_runtime"
date -Is >"$queue_runtime/queue_started_at.txt"

while kill -0 "$old_pid" 2>/dev/null; do
  date -Is >"$queue_runtime/old_still_running_at.txt"
  sleep 30
done

if [ ! -f "$old_runtime/exit_code.txt" ] || [ "$(cat "$old_runtime/exit_code.txt")" != "0" ]; then
  echo 'Old [0,2] run did not finish with exit code 0; paired launch not started.' >"$queue_runtime/blocked.txt"
  exit 3
fi
date -Is >"$queue_runtime/old_completed_at.txt"

bash "$package_root/archive_old.sh" >"$queue_runtime/archive_old.log" 2>&1

if [ "$(git -C "$repo" rev-parse HEAD)" != "74c715515c94fd367aff274871bf9488e95ff6b3" ]; then
  echo 'Source HEAD changed before launch.' >"$queue_runtime/blocked.txt"
  exit 4
fi
if [ -n "$(git -C "$repo" status --short)" ]; then
  echo 'Source worktree is dirty before launch.' >"$queue_runtime/blocked.txt"
  exit 5
fi
for p in "$control_run" "$control_runtime" "$dvac_run" "$dvac_runtime"; do
  if [ -e "$p" ]; then
    echo "Launch target already exists: $p" >"$queue_runtime/blocked.txt"
    exit 6
  fi
done

mkdir -p "$control_runtime" "$dvac_runtime"
cp /root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_preflight/control.resolved.yaml "$control_runtime/resolved.yaml"
cp /root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_preflight/dvac.resolved.yaml "$dvac_runtime/resolved.yaml"
cp /root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_preflight/contract_diff.json "$control_runtime/paired_contract_diff.json"
cp /root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_preflight/contract_diff.json "$dvac_runtime/paired_contract_diff.json"
sha256sum "$control_runtime/resolved.yaml" >"$control_runtime/resolved.sha256"
sha256sum "$dvac_runtime/resolved.yaml" >"$dvac_runtime/resolved.sha256"

node_ip=$(hostname -I | awk '{print $1}')
setsid bash "$package_root/start_ray_head.sh" 0 "$node_ip" 46001 46001 46100 46599 "$control_runtime" >"$control_runtime/ray_head.log" 2>&1 < /dev/null &
control_head_pid=$!
setsid bash "$package_root/start_ray_head.sh" 1 "$node_ip" 47001 47001 47100 47599 "$dvac_runtime" >"$dvac_runtime/ray_head.log" 2>&1 < /dev/null &
dvac_head_pid=$!
printf '%s\n' "$control_head_pid" >"$control_runtime/ray_head.pid"
printf '%s\n' "$dvac_head_pid" >"$dvac_runtime/ray_head.pid"

"$venv/bin/python" - "$node_ip" "$control_runtime" "$dvac_runtime" <<'PY'
import json
import sys
import time
import ray

node_ip, control_runtime, dvac_runtime = sys.argv[1:]
for port, path in ((46001, control_runtime), (47001, dvac_runtime)):
    address = f"{node_ip}:{port}"
    last = None
    for _ in range(90):
        try:
            ray.init(address=address, logging_level="ERROR")
            resources = ray.cluster_resources()
            assert resources.get("GPU") == 1.0, resources
            assert resources.get("CPU") == 18.0, resources
            with open(f"{path}/ray_cluster_resources.json", "w", encoding="utf-8") as f:
                json.dump({"address": address, "resources": resources}, f, indent=2, sort_keys=True)
                f.write("\n")
            ray.shutdown()
            break
        except Exception as exc:
            last = repr(exc)
            try:
                ray.shutdown()
            except Exception:
                pass
            time.sleep(2)
    else:
        raise RuntimeError(f"Ray head {address} did not become ready: {last}")
PY

setsid bash "$package_root/run_one.sh" 0 "$control_config" "$control_run" "$control_runtime" "$control_name" "$node_ip:46001" "$control_head_pid" >"$control_runtime/foreground.log" 2>&1 < /dev/null &
control_pid=$!
setsid bash "$package_root/run_one.sh" 1 "$dvac_config" "$dvac_run" "$dvac_runtime" "$dvac_name" "$node_ip:47001" "$dvac_head_pid" >"$dvac_runtime/foreground.log" 2>&1 < /dev/null &
dvac_pid=$!

printf '%s\n' "$control_pid" >"$control_runtime/wrapper.pid"
printf '%s\n' "$dvac_pid" >"$dvac_runtime/wrapper.pid"
sleep 1
control_pgid=$(ps -o pgid= -p "$control_pid" | tr -d ' ')
dvac_pgid=$(ps -o pgid= -p "$dvac_pid" | tr -d ' ')
printf '%s\n' "$control_pgid" >"$control_runtime/process_group.txt"
printf '%s\n' "$dvac_pgid" >"$dvac_runtime/process_group.txt"

setsid bash "$package_root/paired_resource_monitor.sh" "$control_runtime" "$dvac_runtime" "$control_pgid" "$control_head_pid" "$dvac_pgid" "$dvac_head_pid" "$queue_runtime/paired_resources.csv" >"$queue_runtime/monitor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"$queue_runtime/monitor.pid"

date -Is >"$queue_runtime/pair_launched_at.txt"
{
  echo "source_head=$(git -C "$repo" rev-parse HEAD)"
  echo "control_pid=$control_pid"
  echo "control_pgid=$control_pgid"
  echo "control_ray_head_pgid=$control_head_pid"
  echo "control_ray_address=$node_ip:46001"
  echo "control_gpu=0"
  echo "control_config=$control_config"
  echo "control_run=$control_run"
  echo "dvac_pid=$dvac_pid"
  echo "dvac_pgid=$dvac_pgid"
  echo "dvac_ray_head_pgid=$dvac_head_pid"
  echo "dvac_ray_address=$node_ip:47001"
  echo "dvac_gpu=1"
  echo "dvac_config=$dvac_config"
  echo "dvac_run=$dvac_run"
} >"$queue_runtime/pair_launch_summary.txt"
