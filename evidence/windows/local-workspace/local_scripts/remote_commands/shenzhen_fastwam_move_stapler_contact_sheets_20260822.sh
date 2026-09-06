#!/usr/bin/env bash
set -euo pipefail

ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
BASE=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1/move_stapler_pad
SUCCESS="$BASE/episode3_randomized-false_success-true.mp4"
FAILURE="$BASE/episode5_randomized-false_success-false.mp4"
OUT=/data/chenyiteng/results/dvac-observation/phase-candidates/fastwam-move_stapler_pad-p2-v1

test -x "$ENV/bin/python"
test -s "$SUCCESS"
test -s "$FAILURE"
test ! -e "$OUT"
mkdir -p "$OUT"

"$ENV/bin/python" - "$SUCCESS" "$FAILURE" "$OUT" <<'PY'
import sys
from pathlib import Path

import cv2
import numpy as np


def make_sheet(source: Path, destination: Path) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count < 12 or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid video metadata: frames={frame_count}, fps={fps}")
    indices = np.linspace(0, frame_count - 1, 12).round().astype(int)
    cells = []
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed reading frame {frame_index}: {source}")
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        label = f"frame {int(frame_index)}  t={frame_index / fps:.1f}s"
        cv2.rectangle(frame, (0, 0), (245, 27), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            label,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(frame)
    cap.release()
    sheet = np.vstack([np.hstack(cells[row : row + 4]) for row in range(0, 12, 4)])
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"failed writing contact sheet: {destination}")


success = Path(sys.argv[1])
failure = Path(sys.argv[2])
output = Path(sys.argv[3])
make_sheet(success, output / "success_episode_id4_seed4300003_contact_sheet_12f.png")
make_sheet(failure, output / "failure_episode_id6_seed4300005_contact_sheet_12f.png")
PY

printf 'SOURCE_SUCCESS=%s\n' "$SUCCESS"
printf 'SOURCE_FAILURE=%s\n' "$FAILURE"
find "$OUT" -maxdepth 1 -type f -name '*.png' -printf '%p\t%s\n' | sort
sha256sum "$OUT"/*.png
