set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
run_root=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
runtime_root="${evidence_root}/runtime"
experiment_name=robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_formal_fresh480_v1
config_name=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2_fresh480
expected_head=a85b101bfd905f6d1e0700ae6c3ef1e4fb0ecec4

stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
stage1_manifest_id=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
stage1_manifest_sha256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
norm_stats_sha256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

cd "$repo"
test "$(git rev-parse HEAD)" = "$expected_head"
test "$(git rev-parse '@{upstream}')" = "$expected_head"
test -z "$(git status --porcelain)"
test ! -e "$run_root"
test ! -e "$evidence_root"
test -x "${venv}/bin/python"
test -d "$assets"
test -d "$stage1_model"
test "$(sha256sum "$stage1_manifest" | awk '{print $1}')" = "$stage1_manifest_sha256"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$norm_stats_sha256"
mkdir -p "$runtime_root"

export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RLT_LOG_ROOT="$run_root"
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export RLT_STAGE1_MODEL_PATH="$stage1_model"
export RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
export RLT_STAGE1_MANIFEST_ID="$stage1_manifest_id"
export RLT_STAGE1_MANIFEST_SHA256="$stage1_manifest_sha256"
export RLT_NORM_STATS_SHA256="$norm_stats_sha256"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

overrides=(
  "runner.logger.log_path=${run_root}"
  "runner.logger.experiment_name=${experiment_name}"
)

"${venv}/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name "$config_name" "${overrides[@]}" --cfg job --resolve \
  >"${runtime_root}/resolved.yaml"

"${venv}/bin/python" - "${runtime_root}/resolved.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
assert cfg["runner"]["max_steps"] == 480
assert cfg["runner"]["val_check_interval"] == 25
assert cfg["runner"]["save_interval"] == 25
assert cfg["runner"]["resume_dir"] is None
assert cfg["env"]["train"]["total_num_envs"] == 8
assert cfg["env"]["eval"]["total_num_envs"] == 4
assert cfg["env"]["eval"]["rollout_epoch"] == 5
assert cfg["algorithm"]["rlt_schedule"]["max_updates_per_train_step"] == 1600
assert cfg["algorithm"]["rlt_schedule"]["warmup_min_size"] == 10000
assert cfg["algorithm"]["rlt_schedule"]["warmup_post_collect_updates"] == 30000
assert cfg["algorithm"]["actor_weight_schedule"]["warmup_updates"] == 20000
assert cfg["algorithm"]["actor_weight_schedule"]["ramp_updates"] == 50000
assert cfg["algorithm"]["rlt_dvac"]["mode"] == "apply"
assert cfg["algorithm"]["rlt_dvac"]["selected_l"] == 3
assert cfg["algorithm"]["rlt_dvac"]["strength"] == 0.5
assert cfg["algorithm"]["rlt_dvac"]["z_clip"] == 2.0
print("FORMAL480_RUNTIME_RESOLVED_OK")
PY

{
  printf '%q ' "${venv}/bin/python" -B examples/embodiment/train_embodied_agent.py \
    --config-path "${repo}/examples/embodiment/config" \
    --config-name "$config_name" "${overrides[@]}"
  printf '\n'
} >"${runtime_root}/exact_command.txt"

cp "examples/embodiment/config/${config_name}.yaml" "${runtime_root}/source_overlay.yaml"
{
  printf 'branch\t%s\n' "$(git branch --show-current)"
  printf 'head\t%s\n' "$(git rev-parse HEAD)"
  printf 'upstream\t%s\n' "$(git rev-parse '@{upstream}')"
  printf 'run_root\t%s\n' "$run_root"
  printf 'evidence_root\t%s\n' "$evidence_root"
  printf 'experiment_name\t%s\n' "$experiment_name"
  printf 'config_name\t%s\n' "$config_name"
  printf 'config_sha256\t%s\n' "$(sha256sum "examples/embodiment/config/${config_name}.yaml" | awk '{print $1}')"
  printf 'stage1_model\t%s\n' "$stage1_model"
  printf 'stage1_manifest_sha256\t%s\n' "$stage1_manifest_sha256"
  printf 'norm_stats_sha256\t%s\n' "$norm_stats_sha256"
  printf 'budget_cycles\t480\n'
  printf 'budget_train_episodes\t3840\n'
  printf 'budget_primitive_action_slots_cap\t768000\n'
  printf 'budget_eval_episodes\t400\n'
  printf 'budget_checkpoints\t20\n'
} >"${runtime_root}/run_provenance.tsv"

{
  date -Is
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  for f in memory.current memory.high memory.max memory.events memory.stat memory.pressure; do
    printf '%s\n' "---$f"
    cat "/sys/fs/cgroup/$f"
  done
  df -h /root/autodl-tmp
} >"${runtime_root}/resources_before.txt"

printf 'FORMAL480_PREPARE_OK\n'
cat "${runtime_root}/run_provenance.tsv"
grep -nE 'log_path:|experiment_name:|max_steps:|val_check_interval:|save_interval:|resume_dir:|total_num_envs:|rollout_epoch:|max_episode_steps:|global_batch_size:|micro_batch_size:|max_updates_per_train_step:|warmup_min_size:|warmup_post_collect_updates:|warmup_updates:|ramp_updates:|cache_size:|sample_window_size:|mode: apply|selected_l:|applied_horizon:|z_clip:|strength:|raw_trace_interval_updates:' "${runtime_root}/resolved.yaml"
