set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_pre_smoke_20260729

mkdir -p "$EVIDENCE_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1
export REPO_PATH="$RLT_ROOT"

compose_sft() {
  config_name=$1
  output_name=$2
  export EMBODIED_PATH="$RLT_ROOT/examples/sft"
  "$PYTHON_BIN" -B "$RLT_ROOT/examples/sft/train_vla_sft.py" \
    --config-path "$RLT_ROOT/examples/sft/config" \
    --config-name "$config_name" \
    --cfg job \
    --resolve > "$EVIDENCE_ROOT/$output_name"
}

compose_embodied() {
  config_name=$1
  output_name=$2
  export EMBODIED_PATH="$RLT_ROOT/examples/embodiment"
  "$PYTHON_BIN" -B "$RLT_ROOT/examples/embodiment/train_embodied_agent.py" \
    --config-path "$RLT_ROOT/examples/embodiment/config" \
    --config-name "$config_name" \
    --cfg job \
    --resolve > "$EVIDENCE_ROOT/$output_name"
}

compose_sft \
  robotwin_rlt_stage1_sft_openpi \
  robotwin_rlt_stage1_candidate_resolved.yaml
compose_sft \
  maniskill_rlt_stage1_sft_openpi_pi05 \
  legacy_maniskill_rlt_stage1_resolved.yaml
compose_embodied \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp \
  robotwin_rlt_stage2_candidate_resolved.yaml
compose_embodied \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke \
  robotwin_rlt_stage2_smoke_candidate_resolved.yaml
compose_embodied \
  maniskill_rlt_stage2_ac_mlp \
  legacy_maniskill_rlt_stage2_resolved.yaml
compose_embodied \
  robotwin_adjust_bottle_ppo_openpi \
  legacy_robotwin_pi0_ppo_resolved.yaml
compose_embodied \
  robotwin_adjust_bottle_dsrl_openpi \
  legacy_robotwin_pi0_dsrl_resolved.yaml

wc -l "$EVIDENCE_ROOT"/*_resolved.yaml
sha256sum "$EVIDENCE_ROOT"/*_resolved.yaml
