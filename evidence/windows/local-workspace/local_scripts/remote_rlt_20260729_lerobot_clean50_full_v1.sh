#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

REVISION=9dc9299c163db059931898a9f0852098a61155a1
INTERMEDIATE=/root/autodl-tmp/datasets/robotwin2/intermediate/${REVISION}/adjust_bottle/pi0-aloha-clean50-v1
CANONICAL_PARENT=/root/autodl-tmp/datasets/robotwin2/canonical
REPO_ID=pi0-aloha-clean50-v1
TARGET=${CANONICAL_PARENT}/${REPO_ID}
CONVERTER=/root/autodl-tmp/RoboTwin/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
LOCK=/root/autodl-tmp/datasets/robotwin2/.pi0-aloha-clean50-v1.lock

mkdir -p "$(dirname "$LOCK")" "$CANONICAL_PARENT"
exec 9>"$LOCK"
flock -n 9

printf 'START\t%s\n' "$(date --iso-8601=seconds)"
test -d "$INTERMEDIATE"
test -f "$INTERMEDIATE/rlt_intermediate_validation.json"
test -f "$CONVERTER"
test ! -e "$TARGET"
test ! -L "$TARGET"
printf 'CONVERTER_SHA256\t'
sha256sum "$CONVERTER"

STAGING=$(mktemp -d "${CANONICAL_PARENT}/.${REPO_ID}.hfhome.XXXXXX")
STAGED_TARGET=${STAGING}/${REPO_ID}
printf 'STAGING_HF_HOME\t%s\n' "$STAGING"

export HF_LEROBOT_HOME="$STAGING"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
INTERMEDIATE_ROOT="$INTERMEDIATE" REPO_ID="$REPO_ID" \
  STAGED_TARGET="$STAGED_TARGET" CONVERTER="$CONVERTER" \
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import importlib.util
import os
import random
import re
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "robotwin_lerobot_converter", os.environ["CONVERTER"]
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

intermediate = Path(os.environ["INTERMEDIATE_ROOT"])
target = Path(os.environ["STAGED_TARGET"])
if target.exists():
    raise FileExistsError(target)

def episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode_(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected HDF5 name: {path}")
    return int(match.group(1))

hdf5_files = sorted(intermediate.rglob("episode_*.hdf5"), key=episode_number)
indices = [episode_number(path) for path in hdf5_files]
if indices != list(range(50)):
    raise ValueError(f"expected episode indices 0..49, got {indices}")

random.seed(0)
np.random.seed(0)
dataset = module.create_empty_dataset(
    os.environ["REPO_ID"],
    robot_type="aloha",
    mode="image",
    has_effort=module.has_effort(hdf5_files),
    has_velocity=module.has_velocity(hdf5_files),
    dataset_config=module.DEFAULT_DATASET_CONFIG,
)
module.populate_dataset(
    dataset,
    hdf5_files,
    task="adjust the bottle",
    episodes=list(range(50)),
)
if not target.is_dir():
    raise FileNotFoundError(target)
print("LEROBOT_CONVERTER_OK episodes=50")
PY

TARGET_ROOT="$STAGED_TARGET" INTERMEDIATE_ROOT="$INTERMEDIATE" \
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

target = Path(os.environ["TARGET_ROOT"])
intermediate = Path(os.environ["INTERMEDIATE_ROOT"])
expected = json.loads(
    (intermediate / "rlt_intermediate_validation.json").read_text(encoding="utf-8")
)
expected_rows = {
    int(item["episode_index"]): int(item["rows"]) for item in expected["per_episode"]
}
info_path = target / "meta" / "info.json"
episodes_path = target / "meta" / "episodes.jsonl"
tasks_path = target / "meta" / "tasks.jsonl"
for path in (info_path, episodes_path, tasks_path):
    if not path.is_file():
        raise FileNotFoundError(path)

info = json.loads(info_path.read_text(encoding="utf-8"))
episodes = [
    json.loads(line)
    for line in episodes_path.read_text(encoding="utf-8").splitlines()
    if line
]
tasks = [
    json.loads(line)
    for line in tasks_path.read_text(encoding="utf-8").splitlines()
    if line
]
if info.get("fps") != 50:
    raise ValueError(f"fps={info.get('fps')}")
if info.get("total_episodes") != 50:
    raise ValueError(f"total_episodes={info.get('total_episodes')}")
if info.get("total_frames") != sum(expected_rows.values()):
    raise ValueError(
        f"total_frames={info.get('total_frames')} expected={sum(expected_rows.values())}"
    )
episode_meta_indices = sorted(int(item["episode_index"]) for item in episodes)
if episode_meta_indices != list(range(50)):
    raise ValueError(f"episode metadata indices={episode_meta_indices}")
if not tasks:
    raise ValueError("tasks metadata is empty")

required_columns = {
    "observation.state",
    "action",
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "task_index",
    "episode_index",
    "frame_index",
    "timestamp",
}
per_episode = []
for episode in range(50):
    parquet_path = (
        target / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(parquet_path)
    missing = required_columns.difference(table.column_names)
    if missing:
        raise KeyError(f"episode {episode}: missing {sorted(missing)}")
    rows = int(table.num_rows)
    if rows != expected_rows[episode]:
        raise ValueError(f"episode {episode}: rows={rows} expected={expected_rows[episode]}")
    episode_indices = table["episode_index"].to_pylist()
    frame_indices = table["frame_index"].to_pylist()
    timestamps = table["timestamp"].to_pylist()
    if episode_indices != [episode] * rows:
        raise ValueError(f"episode {episode}: episode_index mismatch")
    if frame_indices != list(range(rows)):
        raise ValueError(f"episode {episode}: frame_index mismatch")
    if not all(
        math.isclose(float(value), index / 50.0, rel_tol=0.0, abs_tol=1e-6)
        for index, value in enumerate(timestamps)
    ):
        raise ValueError(f"episode {episode}: timestamp mismatch")
    for column in ("observation.state", "action"):
        values = np.asarray(table[column].to_pylist(), dtype=np.float32)
        if values.shape != (rows, 14) or not np.isfinite(values).all():
            raise ValueError(f"episode {episode}: {column} shape/finite {values.shape}")
    for column in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        if table[column].null_count != 0:
            raise ValueError(f"episode {episode}: null image refs in {column}")
    per_episode.append({"episode_index": episode, "rows": rows})

validation = {
    "episodes": 50,
    "total_frames": sum(item["rows"] for item in per_episode),
    "fps": 50,
    "per_episode": per_episode,
    "required_columns": sorted(required_columns),
    "state_dim": 14,
    "action_dim": 14,
    "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "task_records": tasks,
    "episode_metadata": episodes,
}
(target / "rlt_canonical_validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({key: validation[key] for key in ("episodes", "total_frames", "fps")}, sort_keys=True))
PY

test ! -e "$TARGET"
mv -T -- "$STAGED_TARGET" "$TARGET"
rmdir "$STAGING"
printf 'CANONICAL_PROMOTED\t%s\n' "$TARGET"
du -sh "$TARGET"
sha256sum \
  "$TARGET/rlt_canonical_validation.json" \
  "$TARGET/meta/info.json" \
  "$TARGET/meta/episodes.jsonl" \
  "$TARGET/meta/tasks.jsonl"
printf 'SUCCESS\t%s\n' "$(date --iso-8601=seconds)"
