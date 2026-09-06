#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$root"

echo '=== existing probe/tool candidates ==='
find evaluations toolkits tests -maxdepth 4 -type f \
  \( -iname '*openpi*' -o -iname '*robotwin*' -o -iname '*parity*' -o -iname '*probe*' \) \
  | sort | sed -n '1,220p'

echo '=== action model relevant symbols ==='
grep -nE 'class OpenPIActionModel|def (sample_actions|predict_action_batch|prepare_inputs|input_transform|output_transform)|return_dvac_telemetry|model_actions|forward_inputs' \
  rlinf/models/embodiment/openpi/openpi_action_model.py

echo '=== builder/load symbols ==='
grep -RInE 'OpenPIActionModel|build.*model|model_path|load.*checkpoint|from_pretrained|load_model|create_model|ModelManager' \
  rlinf evaluations toolkits | sed -n '1,260p'

echo '=== robotwin env symbols ==='
grep -RInE 'class .*RoboTwin|def (reset|load|step|get_obs|create|make)|use_fixed_reset_state_ids|reset_state' \
  rlinf/envs/robotwin evaluations/robotwin | sed -n '1,260p'

echo '=== eval entry ==='
sed -n '1,260p' evaluations/eval_embodied_agent.py

echo '=== current dvac yaml ==='
sed -n '1,280p' evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml

echo 'GPU_RAY_MODEL_SIM_USED=0'
