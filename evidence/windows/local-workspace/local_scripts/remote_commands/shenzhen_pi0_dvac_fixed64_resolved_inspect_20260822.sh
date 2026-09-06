#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RESOLVED=/data/chenyiteng/results/dvac-observation/packets/pi0-adjust_bottle-fixed64-800baf80-v1/resolved.yaml

"$VENV/bin/python" - "$RESOLVED" <<'PY'
import pathlib
import sys
import yaml

cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for key, value in (
    ("placement", cfg["cluster"]["component_placement"]),
    ("total_num_envs", cfg["env"]["eval"]["total_num_envs"]),
    ("rollout_epoch", cfg["env"]["eval"]["rollout_epoch"]),
    ("max_episode_steps", cfg["env"]["eval"]["max_episode_steps"]),
    ("max_steps_per_rollout_epoch", cfg["env"]["eval"]["max_steps_per_rollout_epoch"]),
    ("fixed_reset", cfg["env"]["eval"]["use_fixed_reset_state_ids"]),
    ("telemetry_enabled", cfg["rollout"]["dvac_telemetry"]["enabled"]),
    ("run_id", cfg["rollout"]["dvac_telemetry"]["run_id"]),
    ("source_commit", cfg["rollout"]["dvac_telemetry"]["source_commit"]),
    ("model_path", cfg["rollout"]["model"]["model_path"]),
    ("log_path", cfg["runner"]["logger"]["log_path"]),
):
    print(f"{key}={value!r}")
PY
sha256sum "$RESOLVED"
test ! -e /data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1
printf '%s\n' 'SZ_PI0_DVAC_FIXED64_RESOLVED_INSPECT_OK'
