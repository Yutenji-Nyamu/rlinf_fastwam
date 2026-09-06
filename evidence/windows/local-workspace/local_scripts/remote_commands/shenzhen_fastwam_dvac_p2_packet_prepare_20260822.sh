#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/dvac-observation/packets/fastwam-multitask-p2-3x16-c63dc9b5-v1
test ! -e "$PACKET"
install -d -m 0755 "$PACKET"
printf 'packet=%s\n' "$PACKET"
