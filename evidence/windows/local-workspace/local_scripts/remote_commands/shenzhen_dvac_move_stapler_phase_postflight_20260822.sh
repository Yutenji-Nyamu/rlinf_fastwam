#!/usr/bin/env bash
set -euo pipefail

OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1
PY=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
test -d "$OUTPUT"
export CUDA_VISIBLE_DEVICES=

"$PY" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

root = Path(sys.argv[1])
summary = json.loads((root / "analysis_summary.json").read_text(encoding="utf-8"))
counts = summary["counts"]
assert counts == {
    "sources": 1,
    "groups": 1,
    "episodes": 16,
    "queries": 162,
    "pre_success_queries": 162,
    "post_success_queries_descriptive_only": 0,
    "horizon_rows": 10368,
    "fastwam_executed_action_frame_rows": 3741,
}
group = summary["groups"][0]
assert group["task"] == "move_stapler_pad"
assert group["success_episodes"] == 11 and group["failure_episodes"] == 5
assert len(group["representatives"]) == 2
assert max(summary["identity_checks"].values()) < 1e-10
source = summary["source_inventory"][0]
assert source["all_finite"] is True
assert source["x_next_matches_x_chain"] is True
assert source["max_abs_final_chain_vs_model_action"] == 0.0

queries = pd.read_csv(root / "query_metrics.csv")
phase_queries = queries[queries["phase_source"] == "manual_raw_contact_sheet_before_dvac"]
assert len(queries) == 162 and len(phase_queries) == 16
assert set(phase_queries["episode_id"].astype(int)) == {4, 6}
actions = pd.read_csv(root / "fastwam_action_frame_metrics.csv")
assert len(actions) == 3741
assert (actions["h"] < 24).all()
phase_actions = actions[actions["phase_source"] == "manual_raw_contact_sheet_before_dvac"]

phase_episode = pd.read_csv(root / "phase_episode_metrics.csv")
phase_summary = pd.read_csv(root / "phase_summary.csv")
storyboard = pd.read_csv(root / "storyboard_index.csv")
assert len(storyboard) == 2
assert set(storyboard["status"]) == {"created"}
storyboard_episode_ids = storyboard["episode_key"].str.extract(r"episode(\d+)")[0].astype(int)
assert set(storyboard_episode_ids) == {4, 6}

pngs = sorted(root.rglob("*.png"))
for path in pngs:
    with Image.open(path) as image:
        image.verify()

print(json.dumps({
    "counts": counts,
    "identity_checks": summary["identity_checks"],
    "phase_query_rows": int(len(phase_queries)),
    "phase_action_rows": int(len(phase_actions)),
    "phase_episode_rows": int(len(phase_episode)),
    "phase_summary_rows": int(len(phase_summary)),
    "storyboard_rows": int(len(storyboard)),
    "png_files": len(pngs),
    "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
}, sort_keys=True))
PY

du -sb "$OUTPUT"
find "$OUTPUT" -type f -name '*.png' -printf '%s\t%p\n' | sort -n
printf '%s\n' 'PHASE_POSTFLIGHT_OK'
