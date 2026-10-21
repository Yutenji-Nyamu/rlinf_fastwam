# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Score each native episode prefix once with frozen Qwen and the legacy RM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--legacy-code", type=Path, required=True)
    p.add_argument("--rm-checkpoint", type=Path, required=True)
    p.add_argument("--rm-threshold", type=float, required=True)
    p.add_argument("--physical-gpu", type=int, choices=(4, 5), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=128)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        p.error("CUDA_VISIBLE_DEVICES must equal the one authorized physical GPU")
    a.output.mkdir(parents=True, exist_ok=False)
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
        text=True,
    )
    identity = [
        line
        for line in inventory.splitlines()
        if int(line.split(",")[0]) == a.physical_gpu
    ][0]
    import torch
    from calibration import report
    from qwen_reward import QwenReward, window_frames

    sys.path.insert(0, str(a.legacy_code))
    from rm_inference import SingleTaskReward

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.lower().removeprefix("gpu-") != identity.split(",")[
        1
    ].strip().lower().removeprefix("gpu-"):
        raise RuntimeError("Logical CUDA device does not match the physical GPU UUID")
    model = QwenReward(a.model).to("cuda:0")
    old = SingleTaskReward(checkpoint_path=a.rm_checkpoint).eval().to("cuda:0")
    source = json.loads((a.dataset / "manifest.json").read_text())
    episodes = source["episodes"][: a.episodes]
    rows = []
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (a.output / "scores.jsonl").open("x") as output:
        for ep_index, e in enumerate(episodes):
            history = np.empty((0, 256, 320, 3), dtype=np.uint8)
            with np.load(a.dataset / e["file"], allow_pickle=False) as z:
                # Materialize this trajectory once; never repeatedly decompress it.
                frames, lengths, labels = z["frames"], z["lengths"], z["success_once"]
                refs, progress = z["frame_indices"], z["frame_progress"]
            for chunk, length in enumerate(lengths):
                future = frames[chunk, :length]
                window = window_frames(history, future)
                score = model.score(window, e["instruction"])
                history = window[-8:].copy()
                with torch.inference_mode():
                    old_scores = (
                        old.compute_reward(torch.from_numpy(future))
                        .float()
                        .cpu()
                        .numpy()
                    )
                row = dict(
                    actual_seed=e["actual_seed"],
                    split=e["split"],
                    legacy_rm_split=e["legacy_rm_split"],
                    chunk=chunk,
                    native_success_once=bool(labels[chunk]),
                    final_native_success=e["success"],
                    source=e["source"],
                    chunk_file=e["chunks"][chunk],
                    frame_indices=refs[chunk, :length].tolist(),
                    frame_progress=progress[chunk, :length].tolist(),
                    old_rm_probability=float(old_scores.max()),
                    old_rm_frame_probabilities=old_scores.tolist(),
                    **score,
                )
                # Match the float32 probability sent through the OpenDW protocol.
                row["probability"] = float(np.float32(row["probability"]))
                rows.append(row)
                output.write(json.dumps(row) + "\n")
            output.flush()
            state = {
                "episodes": ep_index + 1,
                "total": len(episodes),
                "windows": len(rows),
                "seconds": time.monotonic() - start,
            }
            (a.output / "progress.json").write_text(json.dumps(state) + "\n")
    result = (
        report(rows, a.rm_threshold)
        if a.episodes == 128
        else {"admissible": False, "smoke_only": True}
    )
    result.update(
        gpu=identity,
        model=str(a.model),
        episodes=len(episodes),
        windows=len(rows),
        seconds=time.monotonic() - start,
        vlm_seconds=sum(r["seconds"] for r in rows),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
