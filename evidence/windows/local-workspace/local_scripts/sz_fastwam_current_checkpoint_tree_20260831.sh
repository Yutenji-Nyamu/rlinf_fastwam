#!/usr/bin/env bash
set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload
printf '%s\n' 'run directories:'
find "$RUN" -maxdepth 5 -type d -printf '%p\n' 2>/dev/null | sort
printf '%s\n' 'checkpoint-like files:'
find "$RUN" -maxdepth 8 -type f \( -iname '*checkpoint*' -o -iname '*rank*' -o -name '*.pt' -o -name '*.json' \) -printf '%s %p\n' 2>/dev/null | sort -n | tail -80
printf '%s\n' 'all top files:'
find "$RUN" -maxdepth 4 -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -80
