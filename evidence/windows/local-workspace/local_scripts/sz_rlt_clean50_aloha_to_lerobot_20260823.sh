#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

REV=9dc9299c163db059931898a9f0852098a61155a1
ROOT=/data/chenyiteng/datasets/robotwin2
INTERMEDIATE=$ROOT/intermediate/$REV/adjust_bottle/pi0-aloha-clean50-v1
PARENT=$ROOT/canonical
REPO_ID=pi0-aloha-clean50-v1
TARGET=$PARENT/$REPO_ID
CONVERTER=$ROOT/tooling/RoboTwin-c3ddfa8b97d5519efa828b075999bd0006778e5e/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
PY=/home/chenyiteng/venvs/rlt-data-convert-lerobot010/bin/python

test -d "$INTERMEDIATE"
test -f "$CONVERTER"
test ! -e "$TARGET"
mkdir -p "$PARENT"
STAGING=$(mktemp -d "$PARENT/.pi0-aloha-clean50-v1.hfhome.XXXXXX")
STAGED=$STAGING/$REPO_ID
export HF_LEROBOT_HOME="$STAGING"

echo "START $(date --iso-8601=seconds)"
INTERMEDIATE_ROOT="$INTERMEDIATE" REPO_ID="$REPO_ID" STAGED_TARGET="$STAGED" CONVERTER="$CONVERTER" "$PY" -B - <<'PY'
import importlib.util
import os
import random
import re
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("robotwin_lerobot_converter", os.environ["CONVERTER"])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

root = Path(os.environ["INTERMEDIATE_ROOT"])
files = sorted(
    root.rglob("episode_*.hdf5"),
    key=lambda p: int(re.fullmatch(r"episode_(\d+)\.hdf5", p.name).group(1)),
)
if len(files) != 50:
    raise ValueError(f"expected 50 hdf5 files, got {len(files)}")
random.seed(0)
np.random.seed(0)
dataset = module.create_empty_dataset(
    os.environ["REPO_ID"],
    robot_type="aloha",
    mode="image",
    has_effort=module.has_effort(files),
    has_velocity=module.has_velocity(files),
    dataset_config=module.DEFAULT_DATASET_CONFIG,
)
module.populate_dataset(dataset, files, task="adjust the bottle", episodes=list(range(50)))
if not Path(os.environ["STAGED_TARGET"]).is_dir():
    raise FileNotFoundError(os.environ["STAGED_TARGET"])
print("LEROBOT_CONVERSION_OK episodes=50")
PY

TARGET_ROOT="$STAGED" "$PY" -B - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(os.environ["TARGET_ROOT"])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
if info.get("total_episodes") != 50 or info.get("total_frames") != 7188 or info.get("fps") != 50:
    raise ValueError(info)
files = sorted((root / "data").rglob("episode_*.parquet"))
if len(files) != 50:
    raise ValueError(f"expected 50 parquet files, got {len(files)}")
rows = 0
for path in files:
    table = pq.read_table(path, columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.shape != action.shape or state.shape[1:] != (14,):
        raise ValueError(f"{path}: state={state.shape}, action={action.shape}")
    rows += table.num_rows
if rows != 7188:
    raise ValueError(f"expected 7188 frames, got {rows}")
(root / "rlt_canonical_validation.json").write_text(
    json.dumps({"episodes": 50, "total_frames": rows, "fps": 50, "state_dim": 14, "action_dim": 14}, indent=2) + "\n",
    encoding="utf-8",
)
print("CANONICAL_OK episodes=50 frames=7188 fps=50 dims=14")
PY

mv -T "$STAGED" "$TARGET"
rmdir "$STAGING"
du -sh "$TARGET"
echo "SUCCESS $(date --iso-8601=seconds)"
