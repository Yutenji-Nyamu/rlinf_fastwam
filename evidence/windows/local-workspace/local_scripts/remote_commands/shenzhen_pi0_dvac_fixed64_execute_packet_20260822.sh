#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/dvac-observation/packets/pi0-adjust_bottle-fixed64-800baf80-v1
exec bash "$PACKET/shenzhen_pi0_dvac_launch_fixed64_gpu0_3_20260822.sh"
