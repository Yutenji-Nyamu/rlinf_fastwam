#!/usr/bin/env bash
set -euo pipefail

revision=9dc9299c163db059931898a9f0852098a61155a1
raw=/root/autodl-tmp/datasets/robotwin2/raw/${revision}/adjust_bottle/contract_ep0
target=/root/autodl-tmp/datasets/robotwin2/intermediate/${revision}/adjust_bottle/pi0-aloha-clean50-contract-ep0-v1
converter=/root/autodl-tmp/RoboTwin/policy/pi0/scripts/process_data.py

printf 'START %s\n' "$(date -Is)"
test -d "$raw"
test -f "$converter"
if [[ -e "$target" ]]; then
  printf 'target already exists: %s\n' "$target" >&2
  exit 1
fi
mkdir -p "$(dirname "$target")"

RAW_ROOT="$raw" TARGET_ROOT="$target" CONVERTER="$converter" \
/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import importlib.util
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "robotwin_process_data", os.environ["CONVERTER"]
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

raw = Path(os.environ["RAW_ROOT"])
target = Path(os.environ["TARGET_ROOT"])
processed = module.data_transform(str(raw), 1, str(target))
if processed != 1:
    raise RuntimeError(f"expected one processed episode, got {processed}")
PY

RAW_ROOT="$raw" TARGET_ROOT="$target" \
/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

raw_root = Path(os.environ["RAW_ROOT"])
target_root = Path(os.environ["TARGET_ROOT"])
raw_hdf5 = raw_root / "data" / "episode0.hdf5"
processed_dir = target_root / "episode_0"
processed_hdf5 = processed_dir / "episode_0.hdf5"
instruction_path = processed_dir / "instructions.json"

if not processed_hdf5.is_file() or not instruction_path.is_file():
    raise FileNotFoundError("processed episode artifacts are incomplete")

with h5py.File(raw_hdf5, "r") as raw, h5py.File(processed_hdf5, "r") as out:
    raw_left = raw["/joint_action/left_arm"][:]
    raw_left_gripper = raw["/joint_action/left_gripper"][:]
    raw_right = raw["/joint_action/right_arm"][:]
    raw_right_gripper = raw["/joint_action/right_gripper"][:]
    raw_state = np.concatenate(
        (
            raw_left,
            raw_left_gripper[:, None],
            raw_right,
            raw_right_gripper[:, None],
        ),
        axis=1,
    ).astype(np.float32)

    qpos = out["/observations/qpos"][:]
    action = out["/action"][:]
    if qpos.shape != (raw_state.shape[0] - 1, 14):
        raise ValueError(f"unexpected qpos shape: {qpos.shape}")
    if action.shape != qpos.shape:
        raise ValueError(f"action/qpos mismatch: {action.shape} vs {qpos.shape}")
    qpos_max_abs = float(np.max(np.abs(qpos - raw_state[:-1])))
    action_max_abs = float(np.max(np.abs(action - raw_state[1:])))
    if qpos_max_abs != 0.0 or action_max_abs != 0.0:
        raise ValueError(
            f"next-state action contract mismatch: qpos={qpos_max_abs}, action={action_max_abs}"
        )

    image_contract = {}
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        dataset = out[f"/observations/images/{camera}"]
        decoded = cv2.imdecode(
            np.frombuffer(dataset[0], np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None or tuple(decoded.shape) != (480, 640, 3):
            raise ValueError(f"bad processed image for {camera}: {None if decoded is None else decoded.shape}")
        image_contract[camera] = {
            "stored_shape": list(dataset.shape),
            "stored_dtype": str(dataset.dtype),
            "decoded_shape": list(decoded.shape),
        }

with instruction_path.open(encoding="utf-8") as handle:
    instruction = json.load(handle)
instructions = instruction.get("instructions")
if not isinstance(instructions, list) or len(instructions) != 100:
    raise ValueError("processed instructions must preserve the 100 seen prompts")

print(
    json.dumps(
        {
            "rows": int(qpos.shape[0]),
            "state_shape": list(qpos.shape),
            "action_shape": list(action.shape),
            "qpos_raw_t_max_abs": qpos_max_abs,
            "action_raw_t_plus_1_max_abs": action_max_abs,
            "images": image_contract,
            "instruction_count": len(instructions),
        },
        indent=2,
    )
)
PY

find "$target" -type f -printf '%s|%p\n' | LC_ALL=C sort
sha256sum \
  "$target/episode_0/episode_0.hdf5" \
  "$target/episode_0/instructions.json"
du -sh "$target"
printf 'SUCCESS %s\n' "$(date -Is)"
