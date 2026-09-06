#!/usr/bin/env bash
set -euo pipefail
date --iso-8601=seconds
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,user,etimes,args --sort=pid | grep -E 'ppo-control-formal100|ppo-dvac-action-adv-fix-w0p5to1p5-formal100|pi05' | grep -v grep || true
