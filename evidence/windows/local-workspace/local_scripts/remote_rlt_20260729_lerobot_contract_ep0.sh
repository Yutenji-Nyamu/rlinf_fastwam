#!/usr/bin/env bash
set -euo pipefail

revision=9dc9299c163db059931898a9f0852098a61155a1
raw=/root/autodl-tmp/datasets/robotwin2/intermediate/${revision}/adjust_bottle/pi0-aloha-clean50-contract-ep0-v1
canonical_parent=/root/autodl-tmp/datasets/robotwin2/canonical
repo_id=pi0-aloha-clean50-contract-ep0-v1
target=${canonical_parent}/${repo_id}
converter=/root/autodl-tmp/RoboTwin/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py

printf 'START %s\n' "$(date -Is)"
test -d "$raw"
test -f "$converter"
if [[ -e "$target" ]]; then
  printf 'target already exists: %s\n' "$target" >&2
  exit 1
fi
mkdir -p "$canonical_parent"

export HF_LEROBOT_HOME="$canonical_parent"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

RAW_ROOT="$raw" REPO_ID="$repo_id" TARGET_ROOT="$target" CONVERTER="$converter" \
/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import importlib.util
import os
import re
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "robotwin_lerobot_converter", os.environ["CONVERTER"]
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

raw = Path(os.environ["RAW_ROOT"])
target = Path(os.environ["TARGET_ROOT"])
if target.exists():
    raise FileExistsError(target)

def episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode_(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected hdf5 name: {path}")
    return int(match.group(1))

hdf5_files = sorted(raw.rglob("episode_*.hdf5"), key=episode_number)
if [episode_number(path) for path in hdf5_files] != [0]:
    raise ValueError(f"contract conversion expected only episode 0: {hdf5_files}")

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
    episodes=[0],
)
if not target.is_dir():
    raise FileNotFoundError(target)
PY

printf '%s\n' '=== canonical_files ==='
find "$target" -maxdepth 5 -type f -printf '%s|%p\n' | LC_ALL=C sort
du -sh "$target"

TARGET_ROOT="$target" \
/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

target = Path(os.environ["TARGET_ROOT"])
info_path = target / "meta" / "info.json"
episodes_path = target / "meta" / "episodes.jsonl"
tasks_path = target / "meta" / "tasks.jsonl"
parquet_path = target / "data" / "chunk-000" / "episode_000000.parquet"

for path in (info_path, episodes_path, tasks_path, parquet_path):
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
table = pq.read_table(parquet_path)

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
missing = required_columns.difference(table.column_names)
if missing:
    raise KeyError(f"missing parquet columns: {sorted(missing)}")
if table.num_rows != 139:
    raise ValueError(f"expected 139 rows, got {table.num_rows}")
if info.get("fps") != 50:
    raise ValueError(f"expected fps=50, got {info.get('fps')}")
if info.get("total_episodes") != 1 or info.get("total_frames") != 139:
    raise ValueError(
        f"unexpected totals: episodes={info.get('total_episodes')} frames={info.get('total_frames')}"
    )

print(
    json.dumps(
        {
            "fps": info["fps"],
            "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
            "columns": table.column_names,
            "episode_meta": episodes,
            "tasks": tasks,
        },
        indent=2,
        ensure_ascii=False,
    )
)
PY

sha256sum \
  "$target/meta/info.json" \
  "$target/meta/tasks.jsonl" \
  "$target/meta/episodes.jsonl" \
  "$target/data/chunk-000/episode_000000.parquet"
printf 'SUCCESS %s\n' "$(date -Is)"
