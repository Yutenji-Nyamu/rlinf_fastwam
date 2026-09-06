set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
config_name=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2_fresh480
config_path=examples/embodiment/config/${config_name}.yaml
stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json

cd "$repo"
test "$(git rev-parse HEAD)" = 513dbcb7f31ebb639afa5267dff218028d07187b
test "$(git branch --show-current)" = codex/rlt-teacher-dvac-weighting
test "$(git status --short)" = "?? $config_path"
test -d "$stage1_model"
test -f "$stage1_manifest"
test -f "$norm_stats"
test "$(sha256sum "$stage1_manifest" | awk '{print $1}')" = 6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = 649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="${repo}"
export ROBOTWIN_PATH="${assets}"
export ROBOTWIN_ASSETS_PATH="${assets}"
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PI0_NORM_STATS_PATH="${norm_stats}"
export RLT_STAGE1_MODEL_PATH="${stage1_model}"
export RLT_STAGE1_MANIFEST_PATH="${stage1_manifest}"
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

resolved=$(mktemp /tmp/rlt_dvac_fresh480_resolved.XXXXXX.yaml)
trap 'rm -f "$resolved"' EXIT
"${venv}/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name "$config_name" --cfg job --resolve >"$resolved"
"${venv}/bin/python" - "$resolved" <<'PY'
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
assert cfg["algorithm"]["rlt_schedule"]["warmup_min_size"] == 10000
assert cfg["algorithm"]["rlt_schedule"]["warmup_post_collect_updates"] == 30000
assert cfg["algorithm"]["rlt_schedule"]["max_updates_per_train_step"] == 1600
assert cfg["algorithm"]["actor_weight_schedule"]["warmup_updates"] == 20000
assert cfg["algorithm"]["actor_weight_schedule"]["ramp_updates"] == 50000
assert cfg["algorithm"]["rlt_dvac"]["mode"] == "apply"
assert cfg["algorithm"]["rlt_dvac"]["selected_l"] == 3
assert cfg["algorithm"]["rlt_dvac"]["strength"] == 0.5
assert cfg["algorithm"]["rlt_dvac"]["z_clip"] == 2.0
assert cfg["algorithm"]["rlt_dvac"]["raw_trace_interval_updates"] == 1000
print("FORMAL480_RESOLVED_CONTRACT_OK")
PY

git add -- "$config_path"
git diff --cached --check
git commit -m "config(rlt): add teacher DVAC fresh 480 formal run"
printf 'COMMIT=%s\n' "$(git rev-parse HEAD)"
git status --short
git rev-list --left-right --count '@{upstream}...HEAD'
