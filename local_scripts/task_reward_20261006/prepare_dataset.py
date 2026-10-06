"""Prepare native sidecar labels with reset-seed grouped train/val/test splits.

Uses native_binary_recorder.py reward_native or binary_terminal exports. No labels are inferred
from episode outcome. Unknown reset and post-first-success latched labels are
excluded. Split first, sample training frames second; held-out splits retain
all eligible observations for early false-success diagnostics.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_indices(row):
    labels, steps = row["native_success"], row["action_steps"]
    if len(labels) != len(steps) or any(type(x) is not bool and x is not None for x in labels):
        raise ValueError("Invalid native labels")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Non-increasing action clock")
    hits = [i for i, value in enumerate(labels) if value is True]
    first = hits[0] if hits else None
    if row["first_success_position"] != first or bool(row["reference_success"]) != bool(hits):
        raise ValueError("First success/outcome metadata disagrees with native labels")
    if first is None and row["collection_action_steps"] < row["max_episode_steps"]:
        raise ValueError("An incomplete episode is not a real failed episode")
    return [i for i, label in enumerate(labels) if label is not None and (first is None or i <= first)]


def assign_splits(records, seed=42, supplied=None):
    groups = defaultdict(list)
    for row in records:
        groups[f"{row['task_name']}::{row['seed']}"] .append(row)
    if supplied is not None:
        if set(supplied) != set(groups) or set(supplied.values()) - {"train", "val", "test"}:
            raise ValueError("Explicit split plan must assign every task::seed group exactly once")
        return dict(supplied)
    rng = random.Random(seed)
    result = {}
    # Grouped stratification uses whether the seed group contains a successful
    # episode. Repeated trials with mixed outcomes stay together.
    for has_success in (False, True):
        keys = sorted(key for key, rows in groups.items() if any(x["reference_success"] for x in rows) == has_success)
        rng.shuffle(keys)
        if not keys:
            continue
        if len(keys) < 3:
            raise ValueError("Need at least three independent seed groups per outcome stratum, or an explicit split plan")
        holdout = max(1, round(len(keys) * 0.2))
        for index, key in enumerate(keys):
            result[key] = "val" if index < holdout else "test" if index < 2 * holdout else "train"
    return result


def load_aligned_images(row, path):
    """Return stored pixels and native-source-index to stored-position map."""
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        images = archive["native_frames"] if row["capture_mode"] == "reward_native" else archive["frames"]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Invalid pixels or frame/label alignment")
    expected_hash = row["native_frames_sha256"] if row["capture_mode"] == "reward_native" else row["array_sha256"]
    if hashlib.sha256(images.tobytes()).hexdigest() != expected_hash:
        raise ValueError("Native frame hash mismatch")
    if row["capture_mode"] == "reward_native":
        if len(images) != len(row["native_success"]):
            raise ValueError("Full native frame/label length mismatch")
        image_position = {index: index for index in range(len(images))}
    else:
        if len(images) != len(row["frame_indices"]):
            raise ValueError("K8 source-index alignment mismatch")
        image_position = {}
        for position, index in enumerate(row["frame_indices"]):
            if not 0 <= index < len(row["native_success"]):
                raise ValueError("Invalid K8 source frame index")
            if index in image_position and not np.array_equal(images[image_position[index]], images[position]):
                raise ValueError("Repeated source index has different pixels")
            image_position.setdefault(index, position)
    return images, image_position, expected_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, action="append", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frames-per-episode", type=int, default=16)
    args = parser.parse_args()
    records, uids = [], set()
    for root in args.capture_dir:
        for path in sorted(root.glob("native-*.json")):
            row = json.loads(path.read_text())
            if row["task_name"] != args.task or row["capture_mode"] not in ("reward_native", "binary_terminal"):
                raise ValueError(f"Task/mode mismatch: {path}")
            if row["episode_uid"] in uids:
                raise ValueError("Duplicate episode UID")
            uids.add(row["episode_uid"])
            valid_indices(row)
            row["record_path"] = str(path.resolve())
            records.append(row)
    if not records:
        raise ValueError("No native reward captures")
    plan = json.loads(args.split_plan.read_text()) if args.split_plan else None
    split_by_group = assign_splits(records, args.seed, plan)
    buckets = {key: [] for key in ("train", "val", "test")}
    episode_records = []
    images_by_split = defaultdict(list)
    for row in records:
        path = Path(row["record_path"])
        group = f"{row['task_name']}::{row['seed']}"
        split = split_by_group[group]
        images, image_position, expected_hash = load_aligned_images(row, path)
        indices = [index for index in valid_indices(row) if index in image_position]
        if split == "train" and len(indices) > args.train_frames_per_episode:
            if args.train_frames_per_episode < 3:
                raise ValueError("Training sampling must preserve first success and preceding negative")
            # Preserve final milestone and its immediate prior frame alongside
            # spread-out history, without consulting any model scores.
            indices = sorted(set(indices[i] for i in np.linspace(0, len(indices)-1, args.train_frames_per_episode-2, dtype=int)) | set(indices[-2:]))
        for index in indices:
            sample = dict(episode_uid=row["episode_uid"], group_id=group, seed=row["seed"],
                          frame_index=index, action_step=row["action_steps"][index],
                          label=int(row["native_success"][index]), reference_success=row["reference_success"],
                          instruction=row["instruction"])
            buckets[split].append(sample)
            images_by_split[split].append(images[image_position[index]])
        episode_records.append(dict(episode_uid=row["episode_uid"], seed=row["seed"], group_id=group,
                                    split=split, reference_success=row["reference_success"],
                                    source_record=str(path), source_record_sha256=sha256(path),
                                    source_frames_sha256=expected_hash, capture_mode=row["capture_mode"],
                                    native_source_shape=row["source_frame_shape"], stored_shape=list(images.shape[1:]),
                                    eligible_frames=len([x for x in valid_indices(row) if x in image_position]),
                                    selected_frames=len(indices)))
    for split, samples in buckets.items():
        if {x["label"] for x in samples} != {0, 1}:
            raise ValueError(f"{split} must contain both completed and not-completed frames")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, task_name=args.task, task_config=args.task_config, split_seed=args.seed,
                    label_contract="native per-frame labels; unknown reset excluded; at most first positive; latched later frames excluded",
                    split_contract="task+reset seed grouped; 60/20/20 approximately; repeated seed trials never cross splits",
                    source="WorldArena/RLinf native labels and episode split, adapted to the existing native sidecar",
                    pixel_contract="reward_native: exact main RGB; binary_terminal: existing PIL BOX 320x256 K8; source indices preserved; model owns final224 preprocessing",
                    split_by_group=split_by_group, episodes=episode_records, splits={})
    for split, samples in buckets.items():
        target = args.output_dir / f"{split}.npz"
        if len({image.shape for image in images_by_split[split]}) != 1:
            raise ValueError("Mixed native image resolutions are not silently resized")
        np.savez_compressed(target, images=np.stack(images_by_split[split]),
                            labels=np.asarray([x["label"] for x in samples], dtype=np.int8),
                            episode_uids=np.asarray([x["episode_uid"] for x in samples], dtype=np.str_),
                            action_steps=np.asarray([x["action_step"] for x in samples], dtype=np.int32))
        manifest["splits"][split] = dict(path=target.name, sha256=sha256(target), samples=samples,
                                       count=len(samples), positives=sum(x["label"] for x in samples),
                                       episodes=len({x["episode_uid"] for x in samples}))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(records), "splits": {k: {n: v[n] for n in ("count", "positives", "episodes")} for k, v in manifest["splits"].items()}}))


if __name__ == "__main__":
    main()
