#!/usr/bin/env bash
set -u
find /data/chenyiteng/results/rlinf-shenzhen/grpo -maxdepth 5 -type f -name resolved.yaml -print | grep -E 'grpo-formal100-current-4gpu128train64eval-ppo-matched-v2|dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3' | sort
