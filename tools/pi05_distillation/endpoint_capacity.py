# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Target one batch on the authorized smoke pair; optionally verify its resume."""

import importlib.util
import json
import sys
import time
from pathlib import Path


def main():
    """Reuse the accepted training launcher, without a second benchmark stack."""
    directory, micro = Path(sys.argv[1]), int(sys.argv[2])
    root = Path(json.loads((directory / "prepared.json").read_text())["source"])
    spec = importlib.util.spec_from_file_location(
        "capacity_queue", root / "tools/pi05_distillation/run_queue.py"
    )
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    label = f"endpoint-m{micro}"

    def event(phase, **fields):
        row = dict(time=time.time(), phase=phase, **fields)
        queue.save(directory / f"capacity-m{micro}-status.json", row)
        print(json.dumps(row), flush=True)

    queue.event = event
    output = queue.O / f"smoke-{label}"
    if "--resume-export" in sys.argv[3:]:
        receipt = queue.read(output / "checkpoint.json")
        assert receipt["step"] in {6, 14}, (
            "Resume six updates or check the saved export"
        )
        phase = "resume" if receipt["step"] == 6 else "export"
        queue.train(label, output, 14, receipt["path"], label=f"{phase}-{label}")
        queue.accept_smoke(label)
        event("RESUME_EXPORT_PASSED", micro=micro)
        return

    config = queue.read(directory / "endpoint.json")
    config.update(batch_size=2 * micro, microbatch_per_rank=micro, log_every=1)
    queue.save(directory / f"{label}.json", config)
    try:
        receipt = queue.train(label, output, 6, label=f"capacity-{label}")
    except RuntimeError:
        log = (directory / f"capacity-{label}.log").read_text()
        if "out of memory" not in log.lower():
            raise
        result = {"micro": micro, "global_batch": 2 * micro, "oom": True}
    else:
        rows = [
            json.loads(s)
            for s in (output / "metrics-rank0.jsonl").read_text().splitlines()
        ]
        assert rows[-1]["step"] == 6, "Capacity trial did not finish six updates"
        seconds = (rows[-1]["time"] - rows[1]["time"]) / (
            rows[-1]["step"] - rows[1]["step"]
        )
        result = {
            "micro": micro,
            "global_batch": 2 * micro,
            "oom": False,
            "seconds_per_update": seconds,
            "windows_per_second": 2 * micro / seconds,
            "peak_gib": max(r["peak_gib"] for r in rows),
            "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
            "checkpoint": receipt,
        }
    queue.save(directory / f"capacity-m{micro}.json", result)
    event("COMPLETE", result=result)


if __name__ == "__main__":
    main()
