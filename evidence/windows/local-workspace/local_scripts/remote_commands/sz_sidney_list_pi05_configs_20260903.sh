#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
find "$WT/examples/embodiment/config" -maxdepth 1 -type f -iname '*pi05*' -printf '%f\n' | sort
