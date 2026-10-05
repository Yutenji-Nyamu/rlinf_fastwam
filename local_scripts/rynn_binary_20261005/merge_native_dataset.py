"""Assemble exact native episode clips; do not choose samples by model score."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--expected-count", type=int, default=32)
    args = p.parse_args()
    rows, arrays = [], {}
    paths = sorted(args.capture_dir.glob("native-*.json"))
    if len(paths) != args.expected_count:
        raise ValueError(f"Expected exactly {args.expected_count} native labels, got {len(paths)}")
    for path in paths:
        row = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            clip = data["frames"].copy()
        if clip.shape != (8, 256, 320, 3) or clip.dtype != np.uint8:
            raise ValueError("Native clip shape/dtype differs")
        if hashlib.sha256(clip.tobytes()).hexdigest() != row["array_sha256"]:
            raise ValueError("Native clip hash mismatch")
        if type(row["reference_success"]) is not bool:
            raise ValueError("Native label must be a boolean")
        if not row["reference_success"] and row["terminal_action_steps"] < row["max_episode_steps"]:
            raise ValueError("Early interruption cannot be labelled physical failure")
        if row["episode_uid"] in arrays:
            raise ValueError("Duplicate episode UID")
        arrays[row["frames_key"]] = clip
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "samples.npz", **arrays)
    receipt = dict(schema_version=1, cases=rows, native_successes=sum(r["reference_success"] for r in rows),
                   native_failures=sum(not r["reference_success"] for r in rows),
                   selection="all episodes from fixed native evaluation; no balancing or score selection")
    (args.output_dir / "cases.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "cases"}))


if __name__ == "__main__":
    main()
