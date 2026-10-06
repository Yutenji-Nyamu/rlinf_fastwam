"""CPU-only lift_pot reset extraction, following the working bell clean50 path.

Read frame zero from all 50 existing episodes. No simulator, planner, new seed
filtering, GPU use or modification of the donor data is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


TASK = "lift_pot"
CAMERAS = ("head_camera", "left_camera", "right_camera")
DEFAULT_SOURCE = Path("/data/chenyiteng/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/lift_pot/clean50-20261002/aloha-agilex_clean_50")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if TASK not in source.parts:
        raise ValueError("Use the explicit lift_pot raw dataset, not another task")
    if output.suffix != ".npz":
        raise ValueError("Reset output must end in .npz")
    receipt_path = output.with_suffix(".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Inspect the existing reset/receipt; never overwrite")
    files = sorted((source / "data").glob("episode*.hdf5"), key=lambda f: int(f.stem[7:]))
    if [int(path.stem[7:]) for path in files] != list(range(50)):
        raise ValueError("Require the complete existing episode0..49 clean50 dataset")
    views, states, instructions, provenance = [], [], [], []
    for path in files:
        instruction_path = source / "instructions" / (path.stem + ".json")
        instruction = json.loads(instruction_path.read_text())["seen"][0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Missing episode seen[0] instruction")
        with h5py.File(path, "r") as handle:
            encoded = [handle[f"observation/{camera}/rgb"][0].tobytes() for camera in CAMERAS]
            if any(b"XPL-RGB1" in raw for raw in encoded):
                raise ValueError("Tagged RGB JPEG requires its official decoder; donor path was untagged")
            # Preserve the established raw->Sidney and bell-reset decoder.
            # These legacy saved arrays are not given an additional BGR/RGB swap.
            row = [cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) for raw in encoded]
            if any(image is None or image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 for image in row):
                raise ValueError("Invalid initial camera JPEG")
            state = handle["joint_action/vector"][0].astype(np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError("Require simultaneous 14D initial control state")
            provenance.append(dict(episode=path.stem, path=str(path), frame=0,
                source_shapes=[list(image.shape) for image in row],
                raw_initial_image_sha256=[hashlib.sha256(image.tobytes()).hexdigest() for image in row],
                initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                instruction_source=str(instruction_path), instruction_sha256=sha(instruction_path)))
            views.append([cv2.resize(image, (256, 256)) for image in row])
            states.append(state)
            instructions.append(instruction)
    data = np.asarray(views, dtype=np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, main_images=data[:, 0], wrist_images=data[:, 1:],
            states=np.asarray(states), instructions=np.asarray(instructions),
            reset_ids=np.asarray([path.stem for path in files]))
    receipt = dict(task=TASK, count=50, frame=0, source=str(source),
        selection="all 50 existing episodes; no additional success, expert or policy filtering",
        state="joint_action/vector at same frame; 12 joint targets + 2 gripper commands",
        cameras=list(CAMERAS), instruction="seen[0] of matching episode",
        decode="cv2.imdecode, no channel permutation; same legacy decoder as working click_bell reset",
        output=str(output), sha256=sha(output), episodes=provenance,
        native_capture_used=False, cpu_only=True,
        caveat="Existing clean50 is an expert dataset; no claim that it is unbiased native policy data")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.source, args.output)
    print(json.dumps({key: value for key, value in receipt.items() if key != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
