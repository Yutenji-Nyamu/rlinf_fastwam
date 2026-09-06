#!/usr/bin/env bash
set -euo pipefail

PROBE=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1/mihomo_subscription_quota_sanitized_20260821.py
printf '%s  %s\n' 27A9F0B6D91F7B193D555853248012E8CDB2D9FF2202379AA1D81262C6B4FDB5 "$PROBE" | sha256sum --check -
sudo -k
sudo -S -p '' /usr/bin/python3 "$PROBE"
sudo -k
