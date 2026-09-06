#!/usr/bin/env bash
set -euo pipefail
F=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf/rlinf/models/embodiment/openpi/openpi_action_model.py
sed -n '982,1160p' "$F"
