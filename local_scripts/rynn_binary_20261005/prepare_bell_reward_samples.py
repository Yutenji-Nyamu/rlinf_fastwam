"""Up to64 native frames: first success, last true-negative and latched-history controls.

Sample selection uses fixed simulator labels, never RM scores. Post-success
latched images are auxiliary/unscored, not current-contact-positive labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def select_candidates(records):
    buckets = {key: [] for key in ("first_success", "near_false", "post_success_latched")}
    for row in records:
        labels = row["native_success"]
        steps = row["action_steps"]
        if len(labels) != len(steps) or any(type(x) is not bool and x is not None for x in labels):
            raise ValueError("Invalid native per-frame label sequence")
        hits = [i for i, value in enumerate(labels) if value is True]
        first = hits[0] if hits else None
        if row["first_success_position"] != first:
            raise ValueError("First-success metadata disagrees with exact label sequence")
        def add(category, index, label, reason):
            buckets[category].append(dict(episode_uid=row["episode_uid"],
                sample_id=f"{category}::{row['episode_uid']}::frame{index}", category=category,
                frame_index=index, action_step=int(steps[index]), label=label,
                simulator_success=labels[index], source=str(row["record_path"]),
                instruction=row["instruction"], reason=reason))
        if first is not None:
            add("first_success", first, 1, "first observed native success milestone at C32 boundary")
        # 'near' means temporally closest prior negative, not measured distance
        # to the bell. In failed episodes it is the last labelled false frame.
        negative = [i for i, value in enumerate(labels) if value is False and (first is None or i < first)]
        if negative:
            add("near_false", negative[-1], 0,
                "last labelled false before first success" if first is not None else "last labelled false of failed episode")
        after = [i for i in hits if first is not None and i > first]
        if after:
            add("post_success_latched", after[-1], -1,
                "episode already succeeded; native success may be latched, so current contact is not labelled")
    quotas = {"first_success": 16, "near_false": 32, "post_success_latched": 16}
    selected = [r for key in quotas for r in buckets[key][:quotas[key]]]
    return selected, {key: len(values) for key, values in buckets.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    records = []
    for path in sorted(args.capture_dir.glob("native-*.json")):
        row = json.loads(path.read_text())
        if row["task_name"] != "click_bell" or row["capture_mode"] != "reward_native":
            raise ValueError("Reward samples must come from click_bell reward_native captures")
        row["record_path"] = path
        records.append(row)
    if len(records) != 32:
        raise ValueError(f"Expected fixed32 native episodes, got {len(records)}")
    selected, available = select_candidates(records)
    if not selected or len(selected) > 64:
        raise ValueError("Expected at most64 selected native frames")
    images, cache = [], {}
    records_by_id = {r["episode_uid"]: r for r in records}
    for sample in selected:
        uid = sample["episode_uid"]
        if uid not in cache:
            record = records_by_id[uid]
            path = record["record_path"].with_suffix(".npz")
            with np.load(path, allow_pickle=False) as data:
                frames = data["native_frames"].copy()
            if hashlib.sha256(frames.tobytes()).hexdigest() != record["native_frames_sha256"]:
                raise ValueError("Native source-frame hash mismatch")
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) != len(record["native_success"]):
                raise ValueError("Native source pixels or label alignment invalid")
            cache[uid] = frames
        image = cache[uid][sample["frame_index"]]
        sample["image_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
        images.append(image)
    if len({x.shape for x in images}) != 1:
        raise ValueError("Mixed source image sizes; do not silently resize the diagnostic")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sample_path = args.output_dir / "samples.npz"
    np.savez_compressed(sample_path, images=np.stack(images),
        instructions=np.asarray([r["instruction"] for r in selected], dtype=np.str_),
        labels=np.asarray([r["label"] for r in selected], dtype=np.int8),
        sample_ids=np.asarray([r["sample_id"] for r in selected], dtype=np.str_),
        categories=np.asarray([r["category"] for r in selected], dtype=np.str_))
    receipt = dict(schema_version=1, native_episodes=32, count=len(selected), available=available,
        selected={key: sum(r["category"] == key for r in selected) for key in available},
        samples_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(), samples=selected,
        selection="fixed lexicographic episode order; quotas16 first-success /32 prior-negative /16 post-success auxiliary",
        primary_accuracy_labels="first_success=1 and near_false=0 only; no reset labels",
        near_false_meaning="temporally last false before success or failed horizon; no geometric-nearness claim",
        first_success_caveat="first C32 observation of native success; exact physical contact may happen inside the32-action chunk",
        post_success_caveat="label=-1 auxiliary; latched episode success is not a current-contact label; do not count its low RM score as a primary false negative")
    (args.output_dir / "samples.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "samples"}))


if __name__ == "__main__":
    main()
