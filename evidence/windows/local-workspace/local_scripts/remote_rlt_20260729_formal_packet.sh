#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON=/root/autodl-tmp/RLinf/.venv/bin/python
CONFIG=${RLT_ROOT}/examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
DATASET=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
MANIFEST=/root/autodl-tmp/datasets/robotwin2/manifests/pi0-aloha-clean50-v1.json
MODEL=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
STATS=${MODEL}/physical-intelligence/robotwin/norm_stats.json
RUN_ROOT=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
EXPORT_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
NAME=robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1
EXPECTED_CONFIG_SHA=c293bc476ec7458c6bfc5c5c59393e48b286f3e12007f3039ccc282e30645a4c
EXPECTED_MANIFEST_SHA=12ce2ed68632e2b18cf96f52b717edec00bcebb6cc0a446f83da1670d81ef86c
EXPECTED_STATS_SHA=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

test ! -e "$RUN_ROOT"
test ! -L "$RUN_ROOT"
test ! -e "$EXPORT_ROOT"
test ! -L "$EXPORT_ROOT"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA"
test "$(sha256sum "$STATS" | awk '{print $1}')" = "$EXPECTED_STATS_SHA"
mkdir -p "$EXPORT_ROOT"

export PYTHONPATH=${RLT_ROOT}:/root/autodl-tmp/RoboTwin_RLinf
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH=${RLT_ROOT}/examples/sft
export REPO_PATH="$RLT_ROOT"
export ROBOTWIN_RLT_CLEAN50_PATH="$DATASET"
export ROBOTWIN_PI0_BASE_PATH="$MODEL"
export ROBOTWIN_PI0_NORM_STATS_PATH="$STATS"
export RLT_LOG_ROOT="$RUN_ROOT"

"$PYTHON" -B "$RLT_ROOT/examples/sft/train_vla_sft.py" \
  --config-path "$RLT_ROOT/examples/sft/config" \
  --config-name robotwin_rlt_stage1_sft_openpi \
  --cfg job \
  --resolve \
  runner.logger.experiment_name="$NAME" \
  > "$EXPORT_ROOT/formal_resolved.yaml"

CONFIG_PATH="$CONFIG" RESOLVED_PATH="$EXPORT_ROOT/formal_resolved.yaml" \
  DATASET_PATH="$DATASET" MODEL_PATH="$MODEL" STATS_PATH="$STATS" \
  RUN_ROOT="$RUN_ROOT" NAME="$NAME" "$PYTHON" -B - <<'PY'
import os
from pathlib import Path

import yaml

source = yaml.safe_load(Path(os.environ["CONFIG_PATH"]).read_text(encoding="utf-8"))
resolved = yaml.safe_load(Path(os.environ["RESOLVED_PATH"]).read_text(encoding="utf-8"))

if "min_lr" in source["actor"]["optim"]:
    raise ValueError("source config contains forbidden absolute min_lr key")
if source["actor"]["optim"].get("min_lr_rate") != 0.1:
    raise ValueError("source min_lr_rate must be 0.1")
if "min_lr" in resolved["actor"]["optim"]:
    raise ValueError("resolved config contains forbidden absolute min_lr key")

assert resolved["runner"]["max_steps"] == 2000
assert resolved["runner"]["save_interval"] == 2000
assert resolved["runner"]["val_check_interval"] == -1
assert resolved["runner"]["logger"]["log_path"] == os.environ["RUN_ROOT"]
assert resolved["runner"]["logger"]["experiment_name"] == os.environ["NAME"]
assert resolved["data"]["train_data_paths"] == [
    {"dataset_path": os.environ["DATASET_PATH"], "weight": 1.0}
]
assert resolved["actor"]["model"]["model_path"] == os.environ["MODEL_PATH"]
assert (
    resolved["actor"]["model"]["openpi_data"]["norm_stats_path"]
    == os.environ["STATS_PATH"]
)
assert resolved["actor"]["micro_batch_size"] == 16
assert resolved["actor"]["global_batch_size"] == 32
assert resolved["actor"]["fsdp_config"]["sharding_strategy"] == "no_shard"
assert resolved["actor"]["fsdp_config"]["use_orig_params"] is True
assert resolved["actor"]["model"]["openpi"]["rlt_train_vla"] is False
assert resolved["actor"]["model"]["openpi"]["rlt_alpha"] == 0.0
assert resolved["actor"]["optim"]["lr"] == 2.5e-5
assert resolved["actor"]["optim"]["total_training_steps"] == 2000
assert resolved["actor"]["optim"]["lr_warmup_steps"] == 100
assert resolved["actor"]["optim"]["lr_scheduler"] == "cosine"
assert resolved["actor"]["optim"]["min_lr_rate"] == 0.1
print("FORMAL_RESOLVED_CONTRACT_OK")
PY

cp -- "$CONFIG" "$EXPORT_ROOT/source_config.yaml"
cp -- "$MANIFEST" "$EXPORT_ROOT/dataset_manifest.json"
sha256sum "$CONFIG" > "$EXPORT_ROOT/source_config.sha256"
sha256sum "$EXPORT_ROOT/formal_resolved.yaml" > "$EXPORT_ROOT/formal_resolved.sha256"
sha256sum "$MANIFEST" > "$EXPORT_ROOT/dataset_manifest.sha256"
sha256sum "$STATS" > "$EXPORT_ROOT/norm_stats.sha256"

{
  printf 'prepared_at\t%s\n' "$(date --iso-8601=seconds)"
  printf 'branch\t%s\n' "$(git -C "$RLT_ROOT" branch --show-current)"
  printf 'prelaunch_head\t%s\n' "$(git -C "$RLT_ROOT" rev-parse HEAD)"
  printf 'dataset\t%s\n' "$DATASET"
  printf 'dataset_manifest\t%s\n' "$MANIFEST"
  printf 'model\t%s\n' "$MODEL"
  printf 'norm_stats\t%s\n' "$STATS"
  printf 'run_root\t%s\n' "$RUN_ROOT"
  printf 'experiment_name\t%s\n' "$NAME"
  printf 'checkpoint\t%s/%s/checkpoints/global_step_2000\n' "$RUN_ROOT" "$NAME"
} > "$EXPORT_ROOT/prelaunch_provenance.tsv"

cat > "$EXPORT_ROOT/exact_command.txt" <<EOF
cd $RLT_ROOT
export PYTHONPATH=$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export ROBOTWIN_RLT_CLEAN50_PATH=$DATASET
export ROBOTWIN_PI0_BASE_PATH=$MODEL
export ROBOTWIN_PI0_NORM_STATS_PATH=$STATS
export RLT_LOG_ROOT=$RUN_ROOT
timeout --signal=TERM --kill-after=120s 64800s $PYTHON -B examples/sft/train_vla_sft.py --config-path $RLT_ROOT/examples/sft/config --config-name robotwin_rlt_stage1_sft_openpi runner.logger.experiment_name=$NAME
EOF

sha256sum \
  "$EXPORT_ROOT/source_config.yaml" \
  "$EXPORT_ROOT/formal_resolved.yaml" \
  "$EXPORT_ROOT/dataset_manifest.json" \
  "$EXPORT_ROOT/exact_command.txt" \
  "$EXPORT_ROOT/prelaunch_provenance.tsv" \
  > "$EXPORT_ROOT/PRELAUNCH_SHA256SUMS"
printf 'PACKET_READY\t%s\n' "$EXPORT_ROOT"
cat "$EXPORT_ROOT/PRELAUNCH_SHA256SUMS"
