"""Bounded recovery without changing evaluation seeds or completed evidence."""

import time
from pathlib import Path


def limits(plan):
    recovery = plan.get("recovery", {})
    result = dict(attempts=recovery.get("attempts", 1),
                  batch_seconds=recovery.get("batch_seconds", 3600),
                  stall_seconds=recovery.get("stall_seconds", 600),
                  barrier_seconds=recovery.get("barrier_seconds", 7500),
                  continue_errors=recovery.get("continue_errors", False))
    assert result["attempts"] in (1, 2)
    assert 600 <= result["stall_seconds"] <= 1800
    assert 3600 <= result["batch_seconds"] <= 10800
    assert result["barrier_seconds"] >= 2 * result["attempts"] * result["batch_seconds"]
    return result


def archive_failed_batch(root, output, attempt):
    root, output = Path(root).resolve(), Path(output)
    assert not output.is_symlink() and output.resolve().parent == root / "batches"
    assert not (output / "done.json").exists(), "Completed evidence must never be retried"
    if not output.exists():
        return None
    destination = root / "failed_attempts" / f"{output.name}-attempt{attempt}-{time.time_ns()}"
    destination.parent.mkdir(exist_ok=True)
    output.rename(destination)
    return str(destination)
