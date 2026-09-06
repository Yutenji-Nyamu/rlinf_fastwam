#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30
ARCHIVE=${PACKET}.failed-preflight-parity-script
test ! -e "$RUN"
test -d "$PACKET"
test ! -e "$ARCHIVE"
find "$PACKET" -maxdepth 1 -type f -printf '%f %s\n' | sort
mv -- "$PACKET" "$ARCHIVE"
printf 'archived_partial_packet=%s\n' "$ARCHIVE"
