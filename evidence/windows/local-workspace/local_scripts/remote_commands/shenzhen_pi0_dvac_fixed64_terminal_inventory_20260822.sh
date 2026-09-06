#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

printf 'launch_manifest:\n'; cat "$RUN/launch_manifest.txt"
printf 'episode_header:\n'; head -n 1 "$(find "$RUN" -type f -name 'episode_index_env_rank*.csv' | sort | head -n 1)"
printf 'query_header:\n'; head -n 1 "$(find "$RUN" -type f -name 'query_index_rollout_rank*.csv' | sort | head -n 1)"
printf 'run_bytes=%s\n' "$(du -sb "$RUN" | awk '{print $1}')"
printf 'files_by_extension:\n'; find "$RUN" -type f | sed 's/.*\.//' | sort | uniq -c
printf 'mp4_files:\n'; find "$RUN" -type f -name '*.mp4' -printf '%s %p\n' | sort
"$VENV/bin/python" - "$RUN" <<'PY'
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
for p in sorted(root.rglob("trace_rollout_rank*.npz")):
    with np.load(p, allow_pickle=False) as data:
        print("npz", p.name, sorted(data.files))
        for key in ("x_chain", "z_endpoint", "final_model_action", "env_action", "robot_state", "timesteps"):
            a = data[key]
            print("array", p.name, key, a.shape, str(a.dtype), bool(np.isfinite(a).all()))
PY
