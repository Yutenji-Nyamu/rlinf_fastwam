set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
DATA_ROOT=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-contract-ep0-v1
MODEL_ROOT=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
STATS_PATH="$MODEL_ROOT/physical-intelligence/robotwin/norm_stats.json"
RUN_ROOT=/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1
S1A_NAME=robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1
S1B_NAME=robotwin_adjust_bottle_rlt_stage1_s1b_formal_batch_1step_v1

cd "$RLT_ROOT"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test -z "$(git status --short)"
test -d "$DATA_ROOT"
test -d "$MODEL_ROOT"
test -f "$STATS_PATH"
test ! -e "$RUN_ROOT/s1a/$S1A_NAME"
test ! -e "$RUN_ROOT/s1b/$S1B_NAME"

mkdir -p "$EVIDENCE_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1
export ROBOTWIN_RLT_CLEAN50_PATH="$DATA_ROOT"
export ROBOTWIN_PI0_BASE_PATH="$MODEL_ROOT"
export ROBOTWIN_PI0_NORM_STATS_PATH="$STATS_PATH"
export JAX_PLATFORMS=cpu

"$PYTHON_BIN" -B "$RLT_ROOT/examples/sft/train_vla_sft.py" \
  --config-path "$RLT_ROOT/examples/sft/config" \
  --config-name robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke \
  --cfg job \
  --resolve \
  "runner.logger.log_path=$RUN_ROOT/s1a" \
  "runner.logger.experiment_name=$S1A_NAME" \
  > "$EVIDENCE_ROOT/s1a_resolved.yaml"

"$PYTHON_BIN" -B "$RLT_ROOT/examples/sft/train_vla_sft.py" \
  --config-path "$RLT_ROOT/examples/sft/config" \
  --config-name robotwin_rlt_stage1_sft_openpi \
  --cfg job \
  --resolve \
  "runner.logger.log_path=$RUN_ROOT/s1b" \
  "runner.logger.experiment_name=$S1B_NAME" \
  runner.max_steps=1 \
  runner.save_interval=-1 \
  > "$EVIDENCE_ROOT/s1b_resolved.yaml"

"$PYTHON_BIN" -B - "$EVIDENCE_ROOT" "$DATA_ROOT" "$MODEL_ROOT" "$STATS_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

evidence = Path(sys.argv[1])
data_root = sys.argv[2]
model_root = sys.argv[3]
stats_path = sys.argv[4]

expected = {
    "s1a_resolved.yaml": {
        "max_steps": 2,
        "save_interval": 2,
        "micro": 1,
        "global": 2,
        "total_training_steps": 2,
        "lr_warmup_steps": 1,
        "log_path": "/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1a",
        "experiment": "robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1",
    },
    "s1b_resolved.yaml": {
        "max_steps": 1,
        "save_interval": -1,
        "micro": 16,
        "global": 32,
        "total_training_steps": 2000,
        "lr_warmup_steps": 100,
        "log_path": "/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1b",
        "experiment": "robotwin_adjust_bottle_rlt_stage1_s1b_formal_batch_1step_v1",
    },
}

summaries = []
for name, want in expected.items():
    cfg = OmegaConf.load(evidence / name)
    got = {
        "max_steps": int(cfg.runner.max_steps),
        "save_interval": int(cfg.runner.save_interval),
        "micro": int(cfg.actor.micro_batch_size),
        "global": int(cfg.actor.global_batch_size),
        "total_training_steps": int(cfg.actor.optim.total_training_steps),
        "lr_warmup_steps": int(cfg.actor.optim.lr_warmup_steps),
        "log_path": str(cfg.runner.logger.log_path),
        "experiment": str(cfg.runner.logger.experiment_name),
    }
    if got != want:
        raise RuntimeError(f"{name}: expected {want!r}, got {got!r}")
    if len(cfg.cluster.component_placement) != 1:
        raise RuntimeError(f"{name}: unexpected placement {cfg.cluster.component_placement}")
    if str(cfg.cluster.component_placement["actor,env,rollout"]) != "0-1":
        raise RuntimeError(f"{name}: expected actor placement 0-1")
    if str(cfg.data.train_data_paths[0].dataset_path) != data_root:
        raise RuntimeError(f"{name}: dataset path did not resolve")
    if str(cfg.actor.model.model_path) != model_root:
        raise RuntimeError(f"{name}: model path did not resolve")
    if str(cfg.actor.model.openpi_data.norm_stats_path) != stats_path:
        raise RuntimeError(f"{name}: stats path did not resolve")
    if bool(cfg.actor.model.openpi.rlt_train_vla):
        raise RuntimeError(f"{name}: pi0 must remain frozen")
    if str(cfg.actor.fsdp_config.sharding_strategy) != "no_shard":
        raise RuntimeError(f"{name}: unexpected sharding strategy")
    summaries.append({"file": name, **got})

print(json.dumps(summaries, indent=2))
PY

cat > "$EVIDENCE_ROOT/exact_commands.txt" <<EOF
S1-A:
cd $RLT_ROOT
ROBOTWIN_RLT_CLEAN50_PATH=$DATA_ROOT \\
ROBOTWIN_PI0_BASE_PATH=$MODEL_ROOT \\
ROBOTWIN_PI0_NORM_STATS_PATH=$STATS_PATH \\
RLT_LOG_ROOT=$RUN_ROOT/s1a \\
$PYTHON_BIN -B examples/sft/train_vla_sft.py \\
  --config-path $RLT_ROOT/examples/sft/config \\
  --config-name robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke \\
  runner.logger.log_path=$RUN_ROOT/s1a \\
  runner.logger.experiment_name=$S1A_NAME

S1-B:
cd $RLT_ROOT
ROBOTWIN_RLT_CLEAN50_PATH=$DATA_ROOT \\
ROBOTWIN_PI0_BASE_PATH=$MODEL_ROOT \\
ROBOTWIN_PI0_NORM_STATS_PATH=$STATS_PATH \\
RLT_LOG_ROOT=$RUN_ROOT/s1b \\
$PYTHON_BIN -B examples/sft/train_vla_sft.py \\
  --config-path $RLT_ROOT/examples/sft/config \\
  --config-name robotwin_rlt_stage1_sft_openpi \\
  runner.logger.log_path=$RUN_ROOT/s1b \\
  runner.logger.experiment_name=$S1B_NAME \\
  runner.max_steps=1 \\
  runner.save_interval=-1

STOP CONDITIONS:
- stop after S1-A global_step_2 is saved and reload-only exits successfully;
- stop S1-A immediately on non-finite loss, OOM, either rank failure, or unexpected pi0 trainable parameters;
- stop after exactly one S1-B optimizer step; no checkpoint;
- on S1-B OOM, allow at most one evidence-driven retry at micro8/global32/accumulation2.
EOF

sha256sum \
  "$EVIDENCE_ROOT/s1a_resolved.yaml" \
  "$EVIDENCE_ROOT/s1b_resolved.yaml" \
  "$EVIDENCE_ROOT/exact_commands.txt"
wc -l \
  "$EVIDENCE_ROOT/s1a_resolved.yaml" \
  "$EVIDENCE_ROOT/s1b_resolved.yaml"
date -Is
