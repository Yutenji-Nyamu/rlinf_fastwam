#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2

test ! -e "$PACKET"
test ! -e "$RUN"
install -d -m 700 "$PACKET"
stat -c 'packet=%n owner=%U group=%G mode=%a' "$PACKET"
printf '%s\n' 'SZ_GRPO_RETRY_V2_PACKET_CREATED'
