#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
evidence=/root/autodl-tmp/tmp/rlt_formal250_validation_20260730
stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json

cd "$repo"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test ! -e "$evidence"
mkdir -p "$evidence"

export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="$repo"
export RLT_LOG_ROOT=/root/autodl-tmp/tmp/rlt_formal250_compose_log_root
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export RLT_STAGE1_MODEL_PATH="$stage1_model"
export RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

"${venv}/bin/python" -m py_compile \
  tests/unit_tests/test_robotwin_seed_partition.py
"${venv}/bin/ruff" check \
  tests/unit_tests/test_robotwin_seed_partition.py
"${venv}/bin/python" -m pytest -q \
  tests/unit_tests/test_robotwin_seed_partition.py \
  tests/unit_tests/test_robotwin_rlt_contract.py

compose() {
  local name=$1
  local output=$2
  "${venv}/bin/python" -B \
    examples/embodiment/train_embodied_agent.py \
    --config-path "${repo}/examples/embodiment/config" \
    --config-name "$name" \
    --cfg job \
    --resolve >"$evidence/$output"
}

compose \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp \
  legacy_base_resolved.yaml
compose \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250 \
  formal_8env250_resolved.yaml
compose \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_resource_smoke \
  resource_smoke_resolved.yaml

EVIDENCE="$evidence" REPO="$repo" "${venv}/bin/python" -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

evidence = Path(os.environ["EVIDENCE"])
repo = Path(os.environ["REPO"])
base = OmegaConf.load(evidence / "legacy_base_resolved.yaml")
formal = OmegaConf.load(evidence / "formal_8env250_resolved.yaml")
smoke = OmegaConf.load(evidence / "resource_smoke_resolved.yaml")

assert base.runner.max_steps == 0
assert base.env.train.total_num_envs == 4
assert base.env.eval.rollout_epoch == 1
assert base.env.eval.auto_reset is True
assert base.env.eval.use_fixed_reset_state_ids is True

assert formal.runner.max_steps == 250
assert formal.runner.val_check_interval == 25
assert formal.runner.save_interval == 25
assert formal.runner.resume_dir is None
assert (
    formal.runner.logger.experiment_name
    == "robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1"
)
assert formal.env.train.total_num_envs == 8
assert formal.env.train.rollout_epoch == 1
assert formal.env.train.auto_reset is False
assert formal.env.train.max_episode_steps == 200
assert formal.env.train.max_steps_per_rollout_epoch == 200
assert formal.env.eval.total_num_envs == 4
assert formal.env.eval.rollout_epoch == 5
assert formal.env.eval.auto_reset is False
assert formal.env.eval.ignore_terminations is True
assert formal.env.eval.group_size == 1
assert formal.env.eval.use_fixed_reset_state_ids is False
assert formal.env.eval.max_episode_steps == 200
assert formal.env.eval.max_steps_per_rollout_epoch == 200

schedule = formal.algorithm.rlt_schedule
assert formal.algorithm.update_epoch == 5
assert schedule.max_updates_per_train_step == 1600
assert schedule.warmup_min_size == 10000
assert schedule.warmup_post_collect_updates == 30000
assert schedule.train_every_transitions == 1
assert formal.algorithm.critic_actor_ratio == 2
weights = formal.algorithm.actor_weight_schedule
assert weights.warmup_updates == 20000
assert weights.ramp_updates == 50000
assert formal.algorithm.reference_dropout_prob == 0.5
assert formal.algorithm.replay_buffer.cache_size == 50000
assert formal.algorithm.replay_buffer.sample_window_size == 50000
assert formal.algorithm.replay_buffer.auto_save is False
assert formal.actor.micro_batch_size == 128
assert formal.actor.global_batch_size == 512
assert formal.actor.model.z_dim == 2048
assert formal.actor.model.proprio_dim == 14
assert formal.actor.model.action_dim == 14
assert formal.actor.model.num_action_chunks == 10
assert formal.actor.model.fixed_std == 0.002
assert formal.rollout.rlt_feature_model.openpi.action_horizon == 50
assert formal.rollout.rlt_feature_model.openpi.action_chunk == 10
assert formal.algorithm.rlt_route.type == "full_task"
assert formal.runner.weight_sync_interval == 1
assert formal.weight_syncer.patch.init_sync.enabled is True

bank_path = Path(formal.env.eval.seeds_path)
assert bank_path == (
    repo
    / "rlinf/envs/robotwin/seeds/"
    "eval_seeds_adjust_bottle_rlt_periodic20_v1.json"
)
bank_bytes = bank_path.read_bytes()
bank = json.loads(bank_bytes)["adjust_bottle"]["success_seeds"]
assert len(bank) == 20
assert len(set(bank)) == 20
print(f"seed_bank_count={len(bank)}")
print(f"seed_bank_unique={len(set(bank))}")
print(f"seed_bank_sha256={hashlib.sha256(bank_bytes).hexdigest()}")

assert smoke.runner.max_steps == 3
assert smoke.runner.val_check_interval == 3
assert smoke.runner.save_interval == 3
assert smoke.runner.resume_dir is None
assert smoke.env.train.total_num_envs == 8
assert smoke.env.eval.total_num_envs == 4
assert smoke.env.eval.rollout_epoch == 5
assert smoke.algorithm.rlt_schedule.warmup_min_size == 1
assert smoke.algorithm.rlt_schedule.warmup_post_collect_updates == 1600
assert smoke.algorithm.rlt_schedule.max_updates_per_train_step == 1600
assert smoke.algorithm.actor_weight_schedule.warmup_updates == 1600
assert smoke.algorithm.actor_weight_schedule.ramp_updates == 1600

for path in evidence.glob("*_resolved.yaml"):
    assert "UNRESOLVED" not in path.read_text()
    print(f"{path.name}_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY

git diff --check
git status --short
printf '%s\n' FORMAL250_CONFIG_VALIDATION_OK
