#!/usr/bin/env bash
set -euo pipefail
F=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf/rlinf/models/embodiment/openpi/openpi_action_model.py
grep -nE "def (sample_actions|predict_action_chunk|infer_action|_preprocess_observation|prepare_state|_preprocess_images|generate_actions)" "$F"
sed -n '700,1050p' "$F"
