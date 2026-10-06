"""Freeze 32 existing lift_pot evaluation seeds, disjoint from RM collection.

This only subsets an existing task seed bank; it never generates new seeds or
calls an expert/planner. The original SFT must be evaluated on this exact bank
before comparing later policy checkpoints.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


TASK = "lift_pot"


def read_entry(path):
    data = json.loads(Path(path).read_text())
    entry = data.get(TASK)
    seeds = entry.get("success_seeds") if isinstance(entry, dict) else None
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError(f"Require an existing unique {TASK}.success_seeds list: {path}")
    return entry, seeds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, rm_capture_seeds, output):
    output = Path(output)
    receipt_path = output.with_suffix(".receipt.json")
    if output.suffix != ".json" or output.exists() or receipt_path.exists():
        raise ValueError("Require fresh JSON and receipt output paths")
    entry, candidates = read_entry(source)
    _, exclusions = read_entry(rm_capture_seeds)
    if len(exclusions) != 128:
        raise ValueError("Require the exact original RM128 collection bank")
    excluded = set(exclusions)
    selected = [seed for seed in candidates if seed not in excluded][:32]
    if len(selected) != 32:
        raise ValueError("Existing bank has fewer than 32 seeds disjoint from RM128")
    new_entry = copy.deepcopy(entry)
    new_entry["success_seeds"] = selected
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({TASK: new_entry}, stream, indent=2)
        stream.write("\n")
    receipt = dict(task=TASK, count=32, selected_seed_ids=selected,
        source=str(source), source_sha256=sha(source), source_count=len(candidates),
        excluded_rm_capture_seeds=str(rm_capture_seeds), excluded_sha256=sha(rm_capture_seeds), excluded_count=128,
        overlap_with_rm_requested_seeds=0, output=str(output), sha256=sha(output),
        selection="first 32 existing task seeds after excluding the exact RM128 bank; no new planner/policy screening",
        independence="requested seed IDs disjoint; simulator fallback actual seed identity is not asserted",
        baseline="Original SFT native C32/384 baseline required on this exact bank; RM128 43.75% is a different set")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rm-capture-seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.rm_capture_seeds, args.output)), flush=True)


if __name__ == "__main__":
    main()
