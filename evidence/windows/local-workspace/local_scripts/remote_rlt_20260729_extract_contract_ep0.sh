#!/usr/bin/env bash
set -euo pipefail

revision=9dc9299c163db059931898a9f0852098a61155a1
zip=/root/autodl-tmp/datasets/robotwin2/source/${revision}/dataset/adjust_bottle/aloha-agilex_clean_50.zip
parent=/root/autodl-tmp/datasets/robotwin2/raw/${revision}/adjust_bottle
target=${parent}/contract_ep0
expected_sha=5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e

printf 'START %s\n' "$(date -Is)"
test -f "$zip"
test "$(sha256sum "$zip" | awk '{print $1}')" = "$expected_sha"
if [[ -e "$target" ]]; then
  printf 'target already exists: %s\n' "$target" >&2
  exit 1
fi

mkdir -p "$parent"
staging=$(mktemp -d "${parent}/.contract_ep0.tmp.XXXXXX")
printf 'STAGING %s\n' "$staging"

unzip -q "$zip" \
  aloha-agilex_clean_50/data/episode0.hdf5 \
  aloha-agilex_clean_50/instructions/episode0.json \
  aloha-agilex_clean_50/_traj_data/episode0.pkl \
  aloha-agilex_clean_50/seed.txt \
  aloha-agilex_clean_50/scene_info.json \
  -d "$staging"

extracted=${staging}/aloha-agilex_clean_50
test -f "$extracted/data/episode0.hdf5"
test -f "$extracted/instructions/episode0.json"
test -f "$extracted/_traj_data/episode0.pkl"
test -f "$extracted/seed.txt"
test -f "$extracted/scene_info.json"

EXTRACTED_ROOT="$extracted" \
/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

root = Path(os.environ["EXTRACTED_ROOT"])
hdf5_path = root / "data" / "episode0.hdf5"
instruction_path = root / "instructions" / "episode0.json"

with h5py.File(hdf5_path, "r") as handle:
    required = (
        "/joint_action/left_arm",
        "/joint_action/left_gripper",
        "/joint_action/right_arm",
        "/joint_action/right_gripper",
        "/observation/head_camera/rgb",
        "/observation/left_camera/rgb",
        "/observation/right_camera/rgb",
    )
    for key in required:
        if key not in handle:
            raise KeyError(f"missing required raw key: {key}")

    arrays = {key: handle[key] for key in required}
    lengths = {key: int(value.shape[0]) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"time-length mismatch: {lengths}")
    time_steps = next(iter(lengths.values()))
    if time_steps < 2:
        raise ValueError(f"episode too short: {time_steps}")
    if arrays["/joint_action/left_arm"].shape[1:] != (6,):
        raise ValueError(arrays["/joint_action/left_arm"].shape)
    if arrays["/joint_action/right_arm"].shape[1:] != (6,):
        raise ValueError(arrays["/joint_action/right_arm"].shape)

    image_contract = {}
    for camera in ("head_camera", "left_camera", "right_camera"):
        dataset = handle[f"/observation/{camera}/rgb"]
        decoded = cv2.imdecode(
            np.frombuffer(dataset[0], np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None:
            raise ValueError(f"cannot decode first image for {camera}")
        image_contract[camera] = {
            "stored_shape": list(dataset.shape),
            "stored_dtype": str(dataset.dtype),
            "decoded_shape": list(decoded.shape),
            "decoded_dtype": str(decoded.dtype),
        }

with instruction_path.open(encoding="utf-8") as handle:
    instruction = json.load(handle)
seen = instruction.get("seen")
if not isinstance(seen, list) or not seen or not all(
    isinstance(item, str) and item for item in seen
):
    raise ValueError("instructions.seen must be a non-empty list of strings")

result = {
    "time_steps_raw": time_steps,
    "expected_processed_rows": time_steps - 1,
    "joint_shapes": {
        key: list(value.shape)
        for key, value in arrays.items()
        if key.startswith("/joint_action/")
    },
    "images": image_contract,
    "seen_instruction_count": len(seen),
    "seen_instruction_first": seen[0],
}
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

mv "$extracted" "$target"
rmdir "$staging"

printf '%s\n' '=== extracted_files ==='
find "$target" -type f -printf '%s|%p\n' | LC_ALL=C sort
sha256sum \
  "$target/data/episode0.hdf5" \
  "$target/instructions/episode0.json" \
  "$target/_traj_data/episode0.pkl" \
  "$target/seed.txt" \
  "$target/scene_info.json"
du -sh "$target"
printf 'SUCCESS %s\n' "$(date -Is)"
