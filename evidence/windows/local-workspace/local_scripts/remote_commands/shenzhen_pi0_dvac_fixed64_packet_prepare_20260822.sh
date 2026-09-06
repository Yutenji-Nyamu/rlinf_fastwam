#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/dvac-observation/packets/pi0-adjust_bottle-fixed64-800baf80-v1

if test -e "$PACKET"; then
  printf 'packet already exists: %s\n' "$PACKET" >&2
  exit 1
fi
install -d -m 0755 "$PACKET"
printf 'packet=%s\n' "$PACKET"
printf '%s\n' 'SZ_PI0_DVAC_FIXED64_PACKET_CREATED'
