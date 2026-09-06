#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

REV=9dc9299c163db059931898a9f0852098a61155a1
ROOT=/data/chenyiteng/datasets/robotwin2
RAW=$ROOT/raw/$REV/adjust_bottle/clean50-v1
PARENT=$ROOT/intermediate/$REV/adjust_bottle
TARGET=$PARENT/pi0-aloha-clean50-v1
CONVERTER=$ROOT/tooling/RoboTwin-c3ddfa8b97d5519efa828b075999bd0006778e5e/policy/pi0/scripts/process_data.py
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

test -d "$RAW"
test -f "$CONVERTER"
test ! -e "$TARGET"
mkdir -p "$PARENT"
STAGING=$(mktemp -d "$PARENT/.pi0-aloha-clean50-v1.XXXXXX")
STAGED=$STAGING/pi0-aloha-clean50-v1

echo "START $(date --iso-8601=seconds)"
RAW_ROOT="$RAW" TARGET_ROOT="$STAGED" CONVERTER="$CONVERTER" "$PY" -B - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

import h5py

spec = importlib.util.spec_from_file_location("robotwin_process_data", os.environ["CONVERTER"])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
processed = module.data_transform(os.environ["RAW_ROOT"], 50, os.environ["TARGET_ROOT"])
if processed != 50:
    raise RuntimeError(f"expected 50 processed episodes, got {processed}")

target = Path(os.environ["TARGET_ROOT"])
rows = 0
for episode in range(50):
    path = target / f"episode_{episode}" / f"episode_{episode}.hdf5"
    with h5py.File(path, "r") as handle:
        qpos = handle["/observations/qpos"]
        action = handle["/action"]
        if qpos.shape != action.shape or qpos.shape[1:] != (14,):
            raise ValueError(f"episode {episode}: qpos={qpos.shape}, action={action.shape}")
        rows += int(qpos.shape[0])
if rows != 7188:
    raise ValueError(f"expected 7188 rows, got {rows}")
(target / "rlt_intermediate_validation.json").write_text(
    json.dumps({"episodes": 50, "total_rows": rows, "state_dim": 14, "action_dim": 14}, indent=2) + "\n",
    encoding="utf-8",
)
print(f"INTERMEDIATE_OK episodes=50 rows={rows}")
PY

mv -T "$STAGED" "$TARGET"
rmdir "$STAGING"
du -sh "$TARGET"
echo "SUCCESS $(date --iso-8601=seconds)"
