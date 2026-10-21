# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Pack existing native captures once, preserving block labels and frame refs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(capture: Path, seed_plan: Path, output: Path) -> dict:
    """Select 128 trajectories before scoring; keep actual seeds in one split."""
    plan = json.loads(seed_plan.read_text())
    excluded = set(plan["policy_eval"])
    legacy_holdout = sorted(plan["heldout"])
    records = {}
    duplicate_seeds = []
    for p in sorted(capture.glob("*/episode.json")):
        e = json.loads(p.read_text())
        seed = int(e["actual_seed"])
        if not e["complete"] or seed in excluded:
            continue
        if seed in records:
            duplicate_seeds.append(seed)
            continue
        if e["task"] != "lift_pot" or e["policy"] != "original_pi05":
            raise ValueError("Calibration requires the initial lift_pot policy capture")
        e.update(source=str(p), actual_seed=seed)
        records[seed] = e
    rng = random.Random(20261021)
    selected = []
    for success in (False, True):
        pool = [e for e in records.values() if bool(e["success"]) == success]
        rng.shuffle(pool)
        # Include old RM holdout episodes in our holdout to retain an independent
        # old-RM diagnostic subset. The full Qwen holdout remains score-blind.
        pool.sort(key=lambda e: e["actual_seed"] not in legacy_holdout)
        if len(pool) < 64:
            raise ValueError("Need 64 unique successful and 64 failed trajectories")
        for split, rows in (("holdout", pool[:32]), ("calibration", pool[32:64])):
            for e in rows:
                e = dict(e, split=split)
                seed = e["actual_seed"]
                e["legacy_rm_split"] = (
                    (
                        "test"
                        if legacy_holdout.index(seed) >= len(legacy_holdout) // 2
                        else "val"
                    )
                    if seed in legacy_holdout
                    else "train"
                )
                selected.append(e)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "selection_seed": 20261021,
        "duplicate_actual_seeds": duplicate_seeds,
        "selection": "64/64 calibration/holdout; 32 success + 32 failure each, before VLM scoring",
        "window": "latest 16 valid future head frames; no duplicated chunk-start frame",
        "timebase": "normalized controller progress within C32, not physical timestamps",
        "source_capture": str(capture),
        "episodes": [],
    }
    for e in selected:
        chunks = e["chunks"]
        frames = np.zeros((len(chunks), 8, 256, 320, 3), dtype=np.uint8)
        lengths = np.zeros(len(chunks), dtype=np.int64)
        success_once = np.zeros(len(chunks), dtype=bool)
        frame_indices = np.full((len(chunks), 8), -1, dtype=np.int64)
        frame_progress = np.full((len(chunks), 8), np.nan, dtype=np.float32)
        seen = False
        for i, chunk in enumerate(chunks):
            path = Path(e["source"]).parent / chunk["file"]
            with np.load(path, allow_pickle=False) as z:
                valid = [j for j in range(1, 9) if not z["image_is_pad"][j]]
                if not valid:
                    raise ValueError(f"No new valid frame in {path}")
                head = z["head"]
                for k, j in enumerate(valid):
                    frames[i, k] = np.asarray(
                        Image.fromarray(head[j]).resize(
                            (320, 256), Image.Resampling.BILINEAR
                        )
                    )
                    frame_indices[i, k] = j
                    frame_progress[i, k] = z["frame_progress"][j]
                lengths[i] = len(valid)
                seen = (
                    seen
                    or bool(z["success"][~z["image_is_pad"]].any())
                    or bool(chunk["success"])
                )
                success_once[i] = seen
        if bool(success_once[-1]) != bool(e["success"]):
            raise ValueError(f"Episode/chunk native labels disagree: {e['source']}")
        name = f"seed_{e['actual_seed']}.npz"
        np.savez_compressed(
            output / name,
            frames=frames,
            lengths=lengths,
            success_once=success_once,
            frame_indices=frame_indices,
            frame_progress=frame_progress,
        )
        manifest["episodes"].append(
            {
                k: e[k]
                for k in [
                    "actual_seed",
                    "source",
                    "split",
                    "instruction",
                    "success",
                    "legacy_rm_split",
                ]
            }
            | {"file": name, "chunks": [c["file"] for c in chunks]}
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--seed-plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    m = prepare(a.capture, a.seed_plan, a.output)
    print(json.dumps({"episodes": len(m["episodes"]), "output": str(a.output)}))
