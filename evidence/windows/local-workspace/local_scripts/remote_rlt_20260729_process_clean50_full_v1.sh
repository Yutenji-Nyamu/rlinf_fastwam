#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

REVISION=9dc9299c163db059931898a9f0852098a61155a1
RAW=/root/autodl-tmp/datasets/robotwin2/raw/${REVISION}/adjust_bottle/clean50-v1
PARENT=/root/autodl-tmp/datasets/robotwin2/intermediate/${REVISION}/adjust_bottle
TARGET=${PARENT}/pi0-aloha-clean50-v1
CONVERTER=/root/autodl-tmp/RoboTwin/policy/pi0/scripts/process_data.py
LOCK=/root/autodl-tmp/datasets/robotwin2/.pi0-aloha-clean50-v1.lock

mkdir -p "$(dirname "$LOCK")" "$PARENT"
exec 9>"$LOCK"
flock -n 9

printf 'START\t%s\n' "$(date --iso-8601=seconds)"
test -d "$RAW"
test -f "$RAW/rlt_raw_validation.json"
test -f "$CONVERTER"
test ! -e "$TARGET"
test ! -L "$TARGET"
printf 'CONVERTER_SHA256\t'
sha256sum "$CONVERTER"

STAGING=$(mktemp -d "${PARENT}/.pi0-aloha-clean50-v1.process.XXXXXX")
STAGED_TARGET=${STAGING}/pi0-aloha-clean50-v1
printf 'STAGING\t%s\n' "$STAGING"

RAW_ROOT="$RAW" TARGET_ROOT="$STAGED_TARGET" CONVERTER="$CONVERTER" \
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "robotwin_process_data", os.environ["CONVERTER"]
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
processed = module.data_transform(
    os.environ["RAW_ROOT"], 50, os.environ["TARGET_ROOT"]
)
if processed != 50:
    raise RuntimeError(f"expected 50 processed episodes, got {processed}")
print("PROCESS_CONVERTER_OK episodes=50")
PY

RAW_ROOT="$RAW" TARGET_ROOT="$STAGED_TARGET" \
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

raw_root = Path(os.environ["RAW_ROOT"])
target_root = Path(os.environ["TARGET_ROOT"])
per_episode = []
global_qpos_max_abs = 0.0
global_action_max_abs = 0.0
for episode in range(50):
    raw_path = raw_root / "data" / f"episode{episode}.hdf5"
    processed_dir = target_root / f"episode_{episode}"
    processed_path = processed_dir / f"episode_{episode}.hdf5"
    instruction_path = processed_dir / "instructions.json"
    for path in (raw_path, processed_path, instruction_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with h5py.File(raw_path, "r") as raw, h5py.File(processed_path, "r") as out:
        raw_state = np.concatenate(
            (
                raw["/joint_action/left_arm"][:],
                np.asarray(raw["/joint_action/left_gripper"][:]).reshape(-1, 1),
                raw["/joint_action/right_arm"][:],
                np.asarray(raw["/joint_action/right_gripper"][:]).reshape(-1, 1),
            ),
            axis=1,
        ).astype(np.float32)
        qpos = out["/observations/qpos"][:]
        action = out["/action"][:]
        expected_shape = (raw_state.shape[0] - 1, 14)
        if qpos.shape != expected_shape or action.shape != expected_shape:
            raise ValueError(
                f"episode {episode}: shapes qpos={qpos.shape} action={action.shape} "
                f"expected={expected_shape}"
            )
        if not np.isfinite(qpos).all() or not np.isfinite(action).all():
            raise ValueError(f"episode {episode}: non-finite state/action")
        qpos_max_abs = float(np.max(np.abs(qpos - raw_state[:-1])))
        action_max_abs = float(np.max(np.abs(action - raw_state[1:])))
        if qpos_max_abs != 0.0 or action_max_abs != 0.0:
            raise ValueError(
                f"episode {episode}: time contract qpos={qpos_max_abs} action={action_max_abs}"
            )
        global_qpos_max_abs = max(global_qpos_max_abs, qpos_max_abs)
        global_action_max_abs = max(global_action_max_abs, action_max_abs)
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            dataset = out[f"/observations/images/{camera}"]
            if int(dataset.shape[0]) != expected_shape[0]:
                raise ValueError(f"episode {episode}: {camera} rows={dataset.shape[0]}")
            for frame in (0, expected_shape[0] - 1):
                decoded = cv2.imdecode(
                    np.frombuffer(dataset[frame], np.uint8), cv2.IMREAD_COLOR
                )
                if decoded is None or tuple(decoded.shape) != (480, 640, 3):
                    raise ValueError(
                        f"episode {episode}: bad {camera} frame {frame}"
                    )

    instruction = json.loads(instruction_path.read_text(encoding="utf-8"))
    instructions = instruction.get("instructions")
    if not isinstance(instructions, list) or not instructions or not all(
        isinstance(item, str) and item for item in instructions
    ):
        raise ValueError(f"episode {episode}: invalid processed instructions")
    per_episode.append({"episode_index": episode, "rows": expected_shape[0]})

validation = {
    "episodes": 50,
    "total_rows": sum(item["rows"] for item in per_episode),
    "per_episode": per_episode,
    "qpos_raw_t_max_abs": global_qpos_max_abs,
    "action_raw_t_plus_1_max_abs": global_action_max_abs,
    "state_shape_tail": [14],
    "action_shape_tail": [14],
    "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "decoded_shape": [480, 640, 3],
}
(target_root / "rlt_intermediate_validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({key: validation[key] for key in ("episodes", "total_rows", "qpos_raw_t_max_abs", "action_raw_t_plus_1_max_abs")}, sort_keys=True))
PY

test ! -e "$TARGET"
mv -T -- "$STAGED_TARGET" "$TARGET"
rmdir "$STAGING"
printf 'INTERMEDIATE_PROMOTED\t%s\n' "$TARGET"
du -sh "$TARGET"
sha256sum "$TARGET/rlt_intermediate_validation.json"
printf 'SUCCESS\t%s\n' "$(date --iso-8601=seconds)"
