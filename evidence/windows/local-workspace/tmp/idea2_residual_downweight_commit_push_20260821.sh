#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
BRANCH=codex/idea2-dvac-residual-downweight

git -C "$ROOT" diff --check
git -C "$ROOT" add -- \
  rlinf/algorithms/dvac_train_weighting.py \
  rlinf/workers/actor/fsdp_actor_worker.py \
  tests/unit_tests/test_dvac_train_weighting.py \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_2step_smoke.yaml

git -C "$ROOT" commit -m "feat: add residual DVAC action weighting"
timeout 60s git -C "$ROOT" push -u personal "$BRANCH"
git -C "$ROOT" rev-parse HEAD
git -C "$ROOT" status --short --branch
