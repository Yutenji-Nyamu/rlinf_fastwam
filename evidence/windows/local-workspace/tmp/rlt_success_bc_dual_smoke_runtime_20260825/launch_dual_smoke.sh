#!/usr/bin/env bash
set -euo pipefail

package_root=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_runtime_package
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
expected_head=5ace6d9792a74b8f8436f40382461e34d47d2aa3

control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v3
control_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v3/runtime
control_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control
control_name=robotwin_adjust_bottle_rlt_single_gpu_control_success_bc_pair_smoke_v3

method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v3
method_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v3/runtime
method_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_fresh480
method_name=robotwin_adjust_bottle_rlt_single_gpu_success_episode_bc_dvac_smoke_v3

pair_runtime=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v3

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --short)"
for path in "$control_run" "$control_runtime" "$method_run" "$method_runtime" "$pair_runtime"; do
  test ! -e "$path"
done
mkdir -p "$control_runtime" "$method_runtime" "$pair_runtime"
cd "$repo"

export PYTHONPATH="$repo:$assets"
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

common_overrides=(
  "runner.max_steps=1"
  "runner.val_check_interval=1"
  "runner.save_interval=1"
  "runner.resume_dir=null"
  "algorithm.rlt_schedule.max_updates_per_train_step=20"
  "algorithm.rlt_schedule.warmup_min_size=2"
  "algorithm.rlt_schedule.warmup_post_collect_updates=8"
  "algorithm.actor_weight_schedule.warmup_updates=4"
  "algorithm.actor_weight_schedule.ramp_updates=8"
)

RLT_LOG_ROOT="$control_run" "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" --config-name "$control_config" \
  "runner.logger.log_path=${control_run}" "runner.logger.experiment_name=${control_name}" \
  "env.train.task_config.save_path=${control_run}/robotwin_data/train" \
  "env.eval.task_config.save_path=${control_run}/robotwin_data/eval" \
  "${common_overrides[@]}" --cfg job --resolve >"$control_runtime/resolved.yaml"

RLT_LOG_ROOT="$method_run" "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" --config-name "$method_config" \
  "runner.logger.log_path=${method_run}" "runner.logger.experiment_name=${method_name}" \
  "env.train.task_config.save_path=${method_run}/robotwin_data/train" \
  "env.eval.task_config.save_path=${method_run}/robotwin_data/eval" \
  "algorithm.rlt_dvac.raw_trace_interval_updates=1" \
  "algorithm.rlt_dvac.raw_trace_queries_per_update=4" \
  "${common_overrides[@]}" --cfg job --resolve >"$method_runtime/resolved.yaml"

"$venv/bin/python" - "$control_runtime/resolved.yaml" "$method_runtime/resolved.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    control = yaml.safe_load(f)
with open(sys.argv[2], encoding="utf-8") as f:
    method = yaml.safe_load(f)

def get(data, path):
    for key in path.split("."):
        data = data[key]
    return data

for path in (
    "cluster.component_placement", "runner.max_steps", "runner.val_check_interval",
    "runner.save_interval", "env.train.total_num_envs", "env.train.rollout_epoch",
    "env.eval.total_num_envs", "env.eval.rollout_epoch", "actor.global_batch_size",
    "actor.micro_batch_size", "algorithm.update_epoch", "algorithm.critic_actor_ratio",
    "algorithm.rlt_schedule", "algorithm.replay_buffer", "algorithm.actor_weight_schedule",
):
    assert get(control, path) == get(method, path), path
assert get(control, "cluster.component_placement") == {"actor, env, rollout": 0}
assert get(control, "runner.max_steps") == 1
assert get(control, "env.train.total_num_envs") == 8
assert get(control, "env.eval.total_num_envs") == 4
assert get(control, "env.eval.rollout_epoch") == 5
assert get(control, "actor.global_batch_size") == 512
assert get(control, "actor.micro_batch_size") == 128
assert get(method, "algorithm.rlt_dvac.application") == "success_episode_bc"
assert get(method, "algorithm.rlt_dvac.strength") == 0.25
assert get(method, "algorithm.rlt_dvac.success_scale") == 1.0
print("DUAL_SINGLE_GPU_SMOKE_CONTRACT_OK")
PY
sha256sum "$control_runtime/resolved.yaml" >"$control_runtime/resolved.sha256"
sha256sum "$method_runtime/resolved.yaml" >"$method_runtime/resolved.sha256"

node_ip=$(hostname -I | awk '{print $1}')
setsid bash "$package_root/start_ray_head.sh" 0 "$node_ip" 48001 48001 48100 48599 "$control_runtime" >"$control_runtime/ray_head.log" 2>&1 < /dev/null &
control_head_pid=$!
setsid bash "$package_root/start_ray_head.sh" 1 "$node_ip" 49001 49001 49100 49599 "$method_runtime" >"$method_runtime/ray_head.log" 2>&1 < /dev/null &
method_head_pid=$!
printf '%s\n' "$control_head_pid" >"$control_runtime/ray_head.pid"
printf '%s\n' "$method_head_pid" >"$method_runtime/ray_head.pid"
cleanup_heads() {
  kill -INT -- "-${control_head_pid}" 2>/dev/null || true
  kill -INT -- "-${method_head_pid}" 2>/dev/null || true
}
trap cleanup_heads ERR

"$venv/bin/python" - "$node_ip" "$control_runtime" "$method_runtime" <<'PY'
import json
import sys
import time
import ray

node_ip, control_runtime, method_runtime = sys.argv[1:]
for port, path in ((48001, control_runtime), (49001, method_runtime)):
    address = f"{node_ip}:{port}"
    last = None
    for _ in range(90):
        try:
            ray.init(address=address, logging_level="ERROR")
            resources = ray.cluster_resources()
            assert resources.get("GPU") == 1.0, resources
            assert resources.get("CPU") == 18.0, resources
            with open(f"{path}/ray_cluster_resources.json", "w", encoding="utf-8") as f:
                json.dump({"address": address, "resources": resources}, f, indent=2)
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
        raise RuntimeError(f"Ray head {address} not ready: {last}")
PY

setsid bash "$package_root/run_one_smoke.sh" 0 "$control_config" "$control_run" "$control_runtime" "$control_name" "$node_ip:48001" "$control_head_pid" control >"$control_runtime/foreground.log" 2>&1 < /dev/null &
control_pid=$!
setsid bash "$package_root/run_one_smoke.sh" 1 "$method_config" "$method_run" "$method_runtime" "$method_name" "$node_ip:49001" "$method_head_pid" method >"$method_runtime/foreground.log" 2>&1 < /dev/null &
method_pid=$!
trap - ERR

printf '%s\n' "$control_pid" >"$control_runtime/wrapper.pid"
printf '%s\n' "$method_pid" >"$method_runtime/wrapper.pid"
sleep 1
control_pgid=$(ps -o pgid= -p "$control_pid" | tr -d ' ')
method_pgid=$(ps -o pgid= -p "$method_pid" | tr -d ' ')
printf '%s\n' "$control_pgid" >"$control_runtime/process_group.txt"
printf '%s\n' "$method_pgid" >"$method_runtime/process_group.txt"

setsid bash "$package_root/paired_resource_monitor.sh" "$control_runtime" "$method_runtime" "$control_pgid" "$control_head_pid" "$method_pgid" "$method_head_pid" "$pair_runtime/paired_resources.csv" >"$pair_runtime/monitor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"$pair_runtime/monitor.pid"

date -Is >"$pair_runtime/launched_at.txt"
{
  echo "source_head=$expected_head"
  echo "control_gpu=0"
  echo "control_pid=$control_pid"
  echo "control_pgid=$control_pgid"
  echo "control_ray_address=$node_ip:48001"
  echo "control_run=$control_run"
  echo "method_gpu=1"
  echo "method_pid=$method_pid"
  echo "method_pgid=$method_pgid"
  echo "method_ray_address=$node_ip:49001"
  echo "method_run=$method_run"
} >"$pair_runtime/launch_summary.txt"
cat "$pair_runtime/launch_summary.txt"
