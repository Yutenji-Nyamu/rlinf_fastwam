#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
source_resolved=/root/autodl-tmp/qam_source_resolved_20260731_formal_v1.yaml
formal_resolved=/root/autodl-tmp/qam_formal_resolved_20260731_v1.yaml
resolved_diff=/root/autodl-tmp/qam_source_to_formal_20260731_v1.diff

cd "$repo"
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin_RLinf"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1

git diff --check
"$venv/bin/python" -m compileall -q \
  rlinf/config.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py
"$venv/bin/python" -m ruff check \
  rlinf/config.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/workers/test_qam_worker_helpers.py
"$venv/bin/python" -m ruff format --check \
  rlinf/config.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/workers/test_qam_worker_helpers.py
"$venv/bin/python" -m pytest -q \
  tests/workers/test_qam_worker_helpers.py \
  tests/algorithms/qam/test_core.py \
  tests/algorithms/qam/test_official_fixture.py

"$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_qam_openpi \
  --cfg job --resolve >"$source_resolved"

"$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_qam_openpi \
  "runner.logger.log_path=/root/autodl-tmp/experiments/qam_formal_20260731_v1" \
  "runner.logger.experiment_name=robotwin_adjust_bottle_qam_formal_20260731_v1" \
  runner.max_steps=500 \
  runner.save_interval=25 \
  runner.resume_dir=null \
  runner.ckpt_path=null \
  algorithm.qam.phase=am_on \
  algorithm.qam.inv_temp=1.0 \
  algorithm.qam.warmup_global_inserts=512 \
  algorithm.qam.q_only_updates_before_am=512 \
  algorithm.qam.min_replay_per_rank=32 \
  algorithm.qam.max_updates_per_step=32 \
  actor.global_batch_size=64 \
  actor.micro_batch_size=32 \
  +actor.fsdp_config.save_full_model_weights=false \
  --cfg job --resolve >"$formal_resolved"

"$venv/bin/python" - "$formal_resolved" <<'PY'
import sys

from omegaconf import OmegaConf

from rlinf.config import validate_cfg

config = validate_cfg(OmegaConf.load(sys.argv[1]))
print(
    "QAM_FORMAL_CONFIG_OK",
    config.runner.max_steps,
    config.algorithm.qam.phase,
    config.algorithm.qam.warmup_global_inserts,
    config.algorithm.qam.q_only_updates_before_am,
    config.algorithm.qam.inv_temp,
    config.actor.global_batch_size,
    config.actor.micro_batch_size,
)
PY

diff -u "$source_resolved" "$formal_resolved" >"$resolved_diff" || test "$?" -eq 1
sha256sum \
  examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml \
  "$source_resolved" \
  "$formal_resolved" \
  "$resolved_diff"
