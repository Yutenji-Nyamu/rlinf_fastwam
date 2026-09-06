#!/usr/bin/env bash
set -euo pipefail

LAUNCHER=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2/shenzhen_grpo_launch_formal100_retry_v2_20260822.sh
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
EXPECTED=5c5cc0b9be56e4c53048a97a6902fe267edeeb63d54b36813294046f6d17d103

test ! -e "$RUN"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED"
bash -n "$LAUNCHER"
bash "$LAUNCHER"
