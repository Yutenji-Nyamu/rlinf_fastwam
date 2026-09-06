#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
norm_stats=$model/physical-intelligence/robotwin/norm_stats.json
stage1_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1
stage1_experiment=robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1
stage1_checkpoint=$stage1_root/$stage1_experiment/checkpoints/global_step_2000
stage1_manifest=$stage1_root/artifacts/stage1_artifact_manifest.json
expected_base=8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_base"
test "$(git -C "$worktree" branch --show-current)" = codex/sz-rlt-dvac-pure-single-gpu
git -C "$worktree" diff --check
test -s "$stage1_checkpoint/actor/model_state_dict/full_weights.pt"
test -s "$stage1_manifest"
test -s "$norm_stats"
test -x "$venv/bin/python"

source "$venv/bin/activate"
export ROBOTWIN_PI0_BASE_PATH="$model"
export ROBOTWIN_PATH="$robotwin"
export ROBOTWIN_ASSETS_PATH="$robotwin"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$worktree"
export EMBODIED_PATH="$worktree/examples/embodiment"
export PYTHONPATH="$worktree:$robotwin${PYTHONPATH:+:$PYTHONPATH}"
export RLT_STAGE1_MODEL_PATH="$stage1_checkpoint"
export RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
export RLT_STAGE1_MANIFEST_ID=sz-rlt-stage1-current-ar-clean50-2k-v1
export RLT_STAGE1_MANIFEST_SHA256
RLT_STAGE1_MANIFEST_SHA256=$(sha256sum "$stage1_manifest" | awk '{print $1}')
export RLT_NORM_STATS_SHA256
RLT_NORM_STATS_SHA256=$(sha256sum "$norm_stats" | awk '{print $1}')
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

"$venv/bin/python" -m compileall -q \
  "$worktree/rlinf/algorithms/rlt/dvac_weighting.py" \
  "$worktree/rlinf/algorithms/rlt/transition.py" \
  "$worktree/rlinf/algorithms/rlt/rollout.py" \
  "$worktree/rlinf/models/embodiment/openpi/openpi_action_model.py" \
  "$worktree/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py"

cd "$worktree"
"$venv/bin/python" -m pytest -q \
  tests/unit_tests/test_rlt_dvac_weighting.py \
  tests/unit_tests/test_robotwin_rlt_current_port.py

tmpdir=$(mktemp -d /data/chenyiteng/results/rlinf-rlt/.rlt-pure-check.XXXXXX)
trap 'rm -rf -- "$tmpdir"' EXIT
for config in \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_dvac_pure04
do
  "$venv/bin/python" examples/embodiment/train_embodied_agent.py \
    --config-path "$worktree/examples/embodiment/config" \
    --config-name "$config" \
    --cfg job --resolve > "$tmpdir/$config.yaml"
done

CONTROL="$tmpdir/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control.yaml" \
PURE="$tmpdir/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_dvac_pure04.yaml" \
"$venv/bin/python" - <<'PY'
import copy
import os
import yaml

with open(os.environ["CONTROL"], encoding="utf-8") as handle:
    control = yaml.safe_load(handle)
with open(os.environ["PURE"], encoding="utf-8") as handle:
    pure = yaml.safe_load(handle)

assert control["runner"]["max_steps"] == pure["runner"]["max_steps"] == 480
assert control["runner"]["val_check_interval"] == 25
assert control["runner"]["save_interval"] == 25
assert control["env"]["train"]["total_num_envs"] == 8
assert control["env"]["eval"]["total_num_envs"] == 4
assert control["env"]["eval"]["rollout_epoch"] == 5
assert control["actor"]["global_batch_size"] == 512
assert control["actor"]["micro_batch_size"] == 256
assert control["algorithm"]["update_epoch"] == 5
assert control["algorithm"]["rlt_schedule"]["warmup_min_size"] == 20000
assert control["algorithm"]["rlt_schedule"]["warmup_post_collect_updates"] == 30000
assert control["algorithm"]["replay_buffer"]["cache_size"] == 80000
assert control["algorithm"]["replay_buffer"]["sample_window_size"] == 80000
# PyYAML's YAML-1.1 loader reads the resolved scalar `off` as False; Hydra's
# OmegaConf runtime retains the intended mode string.
assert control["algorithm"]["rlt_dvac"]["mode"] in ("off", False)
method = pure["algorithm"]["rlt_dvac"]
assert method["mode"] == "apply"
assert method["application"] == "success_episode_bc"
assert method["success_target"] == "reference"
assert method["l_values"] == [2, 3, 4]
assert method["selected_l"] == 3
assert method["applied_horizon"] == 10
assert method["z_clip"] == 2.0
assert method["strength"] == 1.5

for cfg in (control, pure):
    cfg["runner"]["logger"]["experiment_name"] = "paired"
    cfg["algorithm"].pop("rlt_dvac", None)
    cfg["rollout"]["rlt_feature_model"]["openpi"].pop("rlt_dvac_mode", None)
assert control == pure, "Control/Pure resolved configs differ outside method/name"
print("CONFIG_PAIR_MATCHED")
PY

printf 'remote_head=%s\nremote_status_files=%s\nFOCUSED_CHECKS_OK\n' \
  "$(git -C "$worktree" rev-parse HEAD)" \
  "$(git -C "$worktree" status --short | wc -l)"
