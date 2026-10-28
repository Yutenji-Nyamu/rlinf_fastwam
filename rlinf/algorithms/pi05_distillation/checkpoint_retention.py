# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Opt-in retention of complete distillation checkpoint/policy pairs."""

import json
import re
import time
from pathlib import Path


def prune_checkpoints(output, keep_last, keep_steps, *, single_task_smoke=False):
    """Keep recent full resume points and every designated evaluation point.

    Only exact, regular checkpoint/policy pairs in this run are eligible.
    Unpaired files, partial saves, nested directories and symlinks are untouched.
    Call only after atomically publishing the newest complete pair's receipt.
    """
    if keep_last is None:
        return []
    output = Path(output).resolve()
    smoke = single_task_smoke and output.name.startswith("smoke-")
    if not isinstance(keep_last, int) or keep_last < (1 if smoke else 2):
        raise ValueError("Retention requires at least two complete resume points")
    receipt = json.loads((output / "checkpoint.json").read_text())
    pairs = {}
    for checkpoint in output.glob("checkpoint-*.pt"):
        match = re.fullmatch(r"checkpoint-(\d{6})\.pt", checkpoint.name)
        if not match:
            continue
        step = int(match[1])
        policy = output / f"policy-{step:06d}.pt"
        if all(p.is_file() and not p.is_symlink() for p in (checkpoint, policy)):
            pairs[step] = (checkpoint, policy)
    current = receipt["step"]
    if current not in pairs or any(
        Path(receipt[key]).resolve() != expected
        for key, expected in zip(("path", "policy"), pairs[current])
    ):
        raise ValueError("Newest checkpoint receipt is not a complete local pair")
    keep = set(keep_steps) | set(sorted(pairs)[-keep_last:]) | {current}
    deleted = []
    for step in sorted(set(pairs) - keep):
        for path in pairs[step]:
            size = path.stat().st_size
            path.unlink()
            deleted.append({"step": step, "path": str(path), "bytes": size})
    if deleted:
        with (output / "retention.jsonl").open("a") as log:
            log.write(json.dumps({"time": time.time(), "deleted": deleted}) + "\n")
    return deleted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Retain latest two and best complete pairs"
    )
    parser.add_argument("output")
    parser.add_argument("--best-step", type=int, required=True)
    args = parser.parse_args()
    removed = prune_checkpoints(args.output, 2, [args.best_step])
    print(json.dumps({"deleted_files": len(removed), "best_step": args.best_step}))
