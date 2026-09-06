#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-smoke1-2gpu64x4-b1024-noeval-phys23-v1
tail -n 80 "$RUN/runtime/driver.log"
