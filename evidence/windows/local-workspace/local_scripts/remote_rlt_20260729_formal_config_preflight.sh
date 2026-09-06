#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
CONFIG=${RLT_ROOT}/examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
DATASET=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
MANIFEST=/root/autodl-tmp/datasets/robotwin2/manifests/pi0-aloha-clean50-v1.json
RUN_ROOT=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
EXPORT_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
EXPECTED_HEAD=66dc388e464660f0ed6a8c48b7a188731d3dbbbe
EXPECTED_CONFIG_SHA=8340ef4e953877de510da18548d0a69802104b7b2f8218698cd0fb586b49a8f2
EXPECTED_MANIFEST_SHA=12ce2ed68632e2b18cf96f52b717edec00bcebb6cc0a446f83da1670d81ef86c

printf 'START\t%s\n' "$(date --iso-8601=seconds)"
test "$(git -C "$RLT_ROOT" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$RLT_ROOT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$RLT_ROOT" status --porcelain)"
test "$(git -C "$RLT_ROOT" rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA"
test -d "$DATASET"
test ! -e "$RUN_ROOT"
test ! -L "$RUN_ROOT"
test ! -e "$EXPORT_ROOT"
test ! -L "$EXPORT_ROOT"
test -z "$(pgrep -f 'examples/sft/train_vla_sft.py.*robotwin_rlt_stage1' || true)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
df -hT /root/autodl-tmp
printf 'SUCCESS\t%s\n' "$(date --iso-8601=seconds)"
