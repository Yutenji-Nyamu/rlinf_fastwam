#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

REVISION=9dc9299c163db059931898a9f0852098a61155a1
ZIP=/root/autodl-tmp/datasets/robotwin2/source/${REVISION}/dataset/adjust_bottle/aloha-agilex_clean_50.zip
RAW_PARENT=/root/autodl-tmp/datasets/robotwin2/raw/${REVISION}/adjust_bottle
TARGET=${RAW_PARENT}/clean50-v1
EXPECTED_SIZE=298659710
EXPECTED_SHA=5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e
LOCK=/root/autodl-tmp/datasets/robotwin2/.pi0-aloha-clean50-v1.lock

mkdir -p "$(dirname "$LOCK")" "$RAW_PARENT"
exec 9>"$LOCK"
flock -n 9

printf 'START\t%s\n' "$(date --iso-8601=seconds)"
test -f "$ZIP"
test "$(stat -c '%s' -- "$ZIP")" -eq "$EXPECTED_SIZE"
test "$(sha256sum "$ZIP" | awk '{print $1}')" = "$EXPECTED_SHA"
test ! -e "$TARGET"
test ! -L "$TARGET"
unzip -tqq "$ZIP"

ZIP_PATH="$ZIP" /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import os
import re
import zipfile
from pathlib import PurePosixPath

path = os.environ["ZIP_PATH"]
with zipfile.ZipFile(path) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos if not info.is_dir()]
    if len(infos) != 207 or len(names) != 202:
        raise ValueError(
            f"expected 207 entries/202 files, got {len(infos)} entries/{len(names)} files"
        )
    if sum(info.file_size for info in infos) != 450_331_107:
        raise ValueError("unexpected uncompressed byte count")
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in name:
            raise ValueError(f"unsafe archive path: {name}")
        if not name.startswith("aloha-agilex_clean_50/"):
            raise ValueError(f"unexpected archive root: {name}")
    patterns = {
        "pkl": r"aloha-agilex_clean_50/_traj_data/episode(\d+)\.pkl",
        "hdf5": r"aloha-agilex_clean_50/data/episode(\d+)\.hdf5",
        "video": r"aloha-agilex_clean_50/video/episode(\d+)\.mp4",
        "instruction": r"aloha-agilex_clean_50/instructions/episode(\d+)\.json",
    }
    for label, pattern in patterns.items():
        indices = sorted(
            int(match.group(1))
            for name in names
            if (match := re.fullmatch(pattern, name))
        )
        if indices != list(range(50)):
            raise ValueError(f"{label} indices: {indices}")
print("ARCHIVE_CONTRACT_OK entries=207 files=202 episodes=50")
PY

STAGING=$(mktemp -d "${RAW_PARENT}/.clean50-v1.extract.XXXXXX")
printf 'STAGING\t%s\n' "$STAGING"
unzip -q "$ZIP" -d "$STAGING"
EXTRACTED=${STAGING}/aloha-agilex_clean_50
test -d "$EXTRACTED"

EXTRACTED_ROOT="$EXTRACTED" /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

root = Path(os.environ["EXTRACTED_ROOT"])
required_keys = (
    "/joint_action/left_arm",
    "/joint_action/left_gripper",
    "/joint_action/right_arm",
    "/joint_action/right_gripper",
    "/observation/head_camera/rgb",
    "/observation/left_camera/rgb",
    "/observation/right_camera/rgb",
)
episode_rows = []
for episode in range(50):
    required_files = (
        root / "_traj_data" / f"episode{episode}.pkl",
        root / "data" / f"episode{episode}.hdf5",
        root / "video" / f"episode{episode}.mp4",
        root / "instructions" / f"episode{episode}.json",
    )
    for path in required_files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    with h5py.File(required_files[1], "r") as handle:
        for key in required_keys:
            if key not in handle:
                raise KeyError(f"episode {episode}: missing {key}")
        lengths = {key: int(handle[key].shape[0]) for key in required_keys}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode {episode}: time mismatch {lengths}")
        time_steps = next(iter(lengths.values()))
        if time_steps < 2:
            raise ValueError(f"episode {episode}: too short")
        if handle["/joint_action/left_arm"].shape[1:] != (6,):
            raise ValueError(f"episode {episode}: left arm shape")
        if handle["/joint_action/right_arm"].shape[1:] != (6,):
            raise ValueError(f"episode {episode}: right arm shape")
        for key in required_keys[:4]:
            if not np.isfinite(handle[key][:]).all():
                raise ValueError(f"episode {episode}: non-finite {key}")
        for camera in ("head_camera", "left_camera", "right_camera"):
            dataset = handle[f"/observation/{camera}/rgb"]
            for frame in (0, time_steps - 1):
                decoded = cv2.imdecode(
                    np.frombuffer(dataset[frame], np.uint8), cv2.IMREAD_COLOR
                )
                if decoded is None or tuple(decoded.shape) != (240, 320, 3):
                    raise ValueError(
                        f"episode {episode}: bad {camera} frame {frame}: "
                        f"{None if decoded is None else decoded.shape}"
                    )

    instruction = json.loads(required_files[3].read_text(encoding="utf-8"))
    seen = instruction.get("seen")
    if not isinstance(seen, list) or not seen or not all(
        isinstance(item, str) and item for item in seen
    ):
        raise ValueError(f"episode {episode}: invalid instructions.seen")
    episode_rows.append({"episode_index": episode, "raw_steps": time_steps, "rows": time_steps - 1})

validation = {
    "episodes": 50,
    "total_rows": sum(item["rows"] for item in episode_rows),
    "per_episode": episode_rows,
    "camera_names": ["head_camera", "left_camera", "right_camera"],
    "decoded_shape": [240, 320, 3],
    "action_state_dim": 14,
}
(root / "rlt_raw_validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({key: validation[key] for key in ("episodes", "total_rows")}, sort_keys=True))
PY

test ! -e "$TARGET"
mv -T -- "$EXTRACTED" "$TARGET"
rmdir "$STAGING"
printf 'RAW_PROMOTED\t%s\n' "$TARGET"
du -sh "$TARGET"
sha256sum "$TARGET/rlt_raw_validation.json"
printf 'SUCCESS\t%s\n' "$(date --iso-8601=seconds)"
