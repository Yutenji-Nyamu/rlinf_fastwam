#!/usr/bin/env bash
set +e
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s0p5_mb256_warm20k_replay80k_fresh480
run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v1
runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v1/runtime
experiment=robotwin_adjust_bottle_rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v1
ray_address=172.17.0.9:52001
namespace=rlt_dvac_pure_smoke_v1

mkdir -p "$run" "$runtime" "/tmp/raydriver_${namespace}" "/tmp/runtmp_${namespace}"
cd "$repo"
export PYTHONPATH="$repo:$assets"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export RLINF_CODE_WORKING_DIR="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_TMPDIR="/tmp/raydriver_${namespace}"
export TMPDIR="/tmp/runtmp_${namespace}"
export RAY_ADDRESS="$ray_address"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RLT_LOG_ROOT="$run"
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

overrides=(
  "runner.logger.log_path=${run}"
  "runner.logger.experiment_name=${experiment}"
  "runner.max_steps=1"
  "runner.val_check_interval=1000"
  "runner.save_interval=1"
  "runner.resume_dir=null"
  "algorithm.rlt_schedule.max_updates_per_train_step=20"
  "algorithm.rlt_schedule.warmup_min_size=2"
  "algorithm.rlt_schedule.warmup_post_collect_updates=8"
  "algorithm.actor_weight_schedule.warmup_updates=4"
  "algorithm.actor_weight_schedule.ramp_updates=8"
  "algorithm.rlt_dvac.raw_trace_interval_updates=1"
  "algorithm.rlt_dvac.raw_trace_queries_per_update=4"
  "env.train.task_config.save_path=${run}/robotwin_data/train"
  "env.eval.task_config.save_path=${run}/robotwin_data/eval"
)

git rev-parse HEAD >"$runtime/source_head.txt"
printf '%s\n' "$config" >"$runtime/config_name.txt"
printf '%s\n' "$ray_address" >"$runtime/ray_address.txt"
printf '%s\n' "$namespace" >"$runtime/cluster_namespace.txt"
{
  printf '%q ' "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
    --config-path "$repo/examples/embodiment/config" \
    --config-name "$config" "${overrides[@]}"
  printf '\n'
} >"$runtime/exact_command.txt"
date -Is >"$runtime/started_at.txt"

"$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name "$config" "${overrides[@]}" \
  >"$runtime/foreground.log" 2>&1
rc=$?

printf '%s\n' "$rc" >"$runtime/exit_code.txt"
date -Is >"$runtime/finished_at.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >"$runtime/resources_after.txt"
printf 'cgroup_memory_current=' >>"$runtime/resources_after.txt"
cat /sys/fs/cgroup/memory.current >>"$runtime/resources_after.txt"
cat /sys/fs/cgroup/memory.events >>"$runtime/resources_after.txt"

echo SMOKE_EXIT_CODE="$rc"
tail -n 80 "$runtime/foreground.log"
echo CHECKPOINTS
find "$run" -path '*checkpoints/global_step_1*' -maxdepth 9 -type f -printf '%p %s\n' | sort
echo TRACES
find "$run" -type f -name '*.npz' -printf '%p %s\n' | sort
echo FATAL_COUNTS
for pattern in 'CUDA out of memory' 'OutOfMemoryError' 'WorkerCrashedError' 'ActorDiedError' 'NCCL error'; do
  printf '%s=%s ' "$pattern" "$(grep -cF "$pattern" "$runtime/foreground.log" || true)"
done
echo
exit "$rc"
