#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/dvac-observation/packets/fastwam-multitask-p2-3x16-c63dc9b5-v1
exec bash "$PACKET/shenzhen_fastwam_dvac_launch_p2_multitask_3x16_gpu3_20260822.sh"
