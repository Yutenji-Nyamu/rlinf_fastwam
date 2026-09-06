#!/usr/bin/env bash
set -euo pipefail

OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1
PY=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
export CUDA_VISIBLE_DEVICES=

"$PY" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])

outcome = pd.read_csv(root / "outcome_summary.csv")
wanted_outcome = outcome[outcome["metric"].isin([
    "episode_mean_S_std",
    "episode_mean_abs_I_std",
    "episode_mean_abs_R_std",
])][[
    "metric", "success_episodes", "failure_episodes", "success_mean", "failure_mean",
    "success_minus_failure", "bootstrap_ci_low", "bootstrap_ci_high",
]]

phase = pd.read_csv(root / "phase_summary.csv")
wanted_phase = phase[phase["metric"].isin(["mean_S_std", "mean_abs_I_std"])][[
    "phase_axis", "phase_coarse", "phase_task", "query_state_phase", "metric",
    "episodes", "mean", "bootstrap_ci_low", "bootstrap_ci_high",
]]

queries = pd.read_csv(root / "query_metrics.csv")
wanted_queries = queries[queries["episode_id"].isin([4, 6])][[
    "episode_id", "success", "query_idx", "action_slot_start", "executed_length",
    "query_state_phase", "phase_coarse", "phase_task", "S_std_L3",
    "mean_abs_I_std_L3", "mean_abs_R_std_L3",
]].sort_values(["episode_id", "query_idx"])

sensitivity = pd.read_csv(root / "L_sensitivity.csv")

print("OUTCOME_JSON=" + wanted_outcome.to_json(orient="records", double_precision=15))
print("PHASE_JSON=" + wanted_phase.to_json(orient="records", double_precision=15))
print("QUERY_JSON=" + wanted_queries.to_json(orient="records", double_precision=15))
print("L_SENSITIVITY_JSON=" + sensitivity.to_json(orient="records", double_precision=15))
PY

parent=$(dirname "$OUTPUT")
name=$(basename "$OUTPUT")
printf 'TAR_GZIP_STDOUT_BYTES='
tar -C "$parent" -czf - "$name" | wc -c
