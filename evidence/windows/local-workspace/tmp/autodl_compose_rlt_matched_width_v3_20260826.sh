#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
audit=/root/autodl-tmp/experiment_exports/rlt_matched_width_formal480_v3_preflight_20260826
control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_matched_width_formal480_20260826_v3
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
mkdir -p "$audit"
cd "$repo"

export PYTHONPATH="$repo:$assets"
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export RLINF_CODE_WORKING_DIR="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export RLT_LOG_ROOT="$control_run"
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

compose_one() {
  local config=$1 run=$2 name=$3 out=$4
  RLT_LOG_ROOT="$run" "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
    --config-path "$repo/examples/embodiment/config" \
    --config-name "$config" \
    "runner.logger.log_path=$run" \
    "runner.logger.experiment_name=$name" \
    "runner.max_steps=480" \
    "runner.val_check_interval=25" \
    "runner.save_interval=25" \
    "runner.resume_dir=null" \
    "env.train.task_config.save_path=$run/robotwin_data/train" \
    "env.eval.task_config.save_path=$run/robotwin_data/eval" \
    --cfg job --resolve >"$out"
}

compose_one \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_mb256_warm20k_replay80k_control \
  "$control_run" \
  robotwin_adjust_bottle_rlt_single_gpu_control_matched_width_formal480_20260826_v3 \
  "$audit/control_resolved.yaml"

compose_one \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_mb256_warm20k_replay80k_gpu1_fresh480 \
  "$method_run" \
  robotwin_adjust_bottle_rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3 \
  "$audit/method_resolved.yaml"

sha256sum "$audit/control_resolved.yaml" "$audit/method_resolved.yaml" >"$audit/resolved.sha256"

"$venv/bin/python" - "$audit/control_resolved.yaml" "$audit/method_resolved.yaml" <<'PY'
import sys, yaml

def flatten(x, p=''):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items(): out.update(flatten(v, f'{p}.{k}' if p else str(k)))
    elif isinstance(x, list):
        for i, v in enumerate(x): out.update(flatten(v, f'{p}[{i}]'))
    else: out[p] = x
    return out

a = flatten(yaml.safe_load(open(sys.argv[1], encoding='utf-8')))
b = flatten(yaml.safe_load(open(sys.argv[2], encoding='utf-8')))
diff = [(k, a.get(k, '<MISSING>'), b.get(k, '<MISSING>')) for k in sorted(a.keys() | b.keys()) if a.get(k, '<MISSING>') != b.get(k, '<MISSING>')]
print('CONTROL_METHOD_DIFF_COUNT', len(diff))
for row in diff: print(repr(row))
required = {
    'actor.micro_batch_size': 256,
    'actor.global_batch_size': 512,
    'env.train.total_num_envs': 8,
    'env.eval.total_num_envs': 4,
    'env.eval.rollout_epoch': 5,
    'runner.max_steps': 480,
    'runner.val_check_interval': 25,
    'runner.save_interval': 25,
    'algorithm.update_epoch': 5,
    'algorithm.rlt_schedule.max_updates_per_train_step': 1600,
    'algorithm.rlt_schedule.warmup_min_size': 20000,
    'algorithm.rlt_schedule.warmup_post_collect_updates': 30000,
    'algorithm.replay_buffer.cache_size': 80000,
    'algorithm.replay_buffer.sample_window_size': 80000,
    'algorithm.actor_weight_schedule.warmup_updates': 20000,
    'algorithm.actor_weight_schedule.ramp_updates': 50000,
    'algorithm.critic_actor_ratio': 2,
}
for label, f in [('control', a), ('method', b)]:
    bad = {k:(f.get(k), v) for k,v in required.items() if f.get(k) != v}
    print(label.upper() + '_REQUIRED_BAD', bad)
print('METHOD_CONTRACT', {k:b.get(k) for k in sorted(b) if k.startswith('algorithm.rlt_dvac') or k.endswith('openpi.rlt_dvac_mode')})
PY

echo SOURCE_STATUS
git status --short
echo OLD_RAY
ps -o pid,ppid,pgid,stat,cmd -p 585779 || true
ss -ltnp | grep ':52001' || true
