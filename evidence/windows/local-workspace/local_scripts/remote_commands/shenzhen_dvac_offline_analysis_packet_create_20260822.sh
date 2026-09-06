#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1
test ! -e "$PACKET"
install -d -m 0755 "$PACKET"
test -d "$PACKET"
printf 'PACKET_CREATED=%s\n' "$PACKET"
